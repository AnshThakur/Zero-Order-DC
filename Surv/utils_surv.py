import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from opacus.accountants import RDPAccountant


def evaluate_coxph(X_syn, T_syn, E_syn, dur_val, evt_val, X_val):
    # Convert torch tensor to numpy if needed
    if isinstance(X_syn, torch.Tensor):
        X_syn = X_syn.detach().cpu().numpy()
    
    # Generate feature names
    n_features = X_syn.shape[1]
    feature_names = [f"x{i}" for i in range(n_features)]

    # Create synthetic training dataframe
    syn_df = pd.DataFrame(X_syn, columns=feature_names)
    syn_df['duration'] = T_syn
    syn_df['event'] = E_syn

    # Train Cox Proportional Hazards model
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(syn_df, duration_col='duration', event_col='event')

    # Prepare validation set
    val_idx = (~np.isnan(dur_val)) & (~np.isinf(dur_val)) & (evt_val >= 0)
    val_df = pd.DataFrame(X_val[val_idx], columns=feature_names)

    # Predict partial hazards
    risk_scores = cph.predict_partial_hazard(val_df).values

    # Compute C-index (higher risk ⇒ shorter survival, so we use -risk)
    c_index = concordance_index(
        event_times=dur_val[val_idx],
        predicted_scores=-risk_scores,
        event_observed=evt_val[val_idx]
    )

    return c_index



def stratified_oversample(real_indices_by_stratum, total_target, generator):
    all_real_indices = []
    weights = []
    for group in real_indices_by_stratum:
        weights.append(len(group))
        all_real_indices.append(group)

    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / (weights.sum() + 1e-9)

    group_sizes = torch.round(weights * total_target).int()
    group_sizes[-1] = total_target - group_sizes[:-1].sum()  # ensure exact total

    sampled_indices = []
    for group, size in zip(all_real_indices, group_sizes):
        if len(group) == 0:
            continue
        group_tensor = group.clone() if isinstance(group, torch.Tensor) else torch.tensor(group)
        selected = group_tensor[torch.randint(0, len(group), (size,), generator=generator)]
        sampled_indices.append(selected)

    return torch.cat(sampled_indices)


def stratify_syn_pred(T_syn_tensor,E_syn_tensor,pred_syn):
    q1, q2 = torch.quantile(T_syn_tensor, torch.tensor([0.33, 0.66]))
    evt_mask = E_syn_tensor == 1
    cens_mask = E_syn_tensor == 0
    idx_early = (evt_mask & (T_syn_tensor <= q1)).nonzero(as_tuple=True)[0]
    idx_mid = (evt_mask & (T_syn_tensor > q1) & (T_syn_tensor <= q2)).nonzero(as_tuple=True)[0]
    idx_late = (evt_mask & (T_syn_tensor > q2)).nonzero(as_tuple=True)[0]
    idx_cens = cens_mask.nonzero(as_tuple=True)[0]

    pred_groups_syn = [pred_syn[idx_early], pred_syn[idx_mid], pred_syn[idx_late], pred_syn[idx_cens]]
    pred_syn_cat = torch.cat(pred_groups_syn, dim=0)
    return pred_syn_cat,q1,q2


def stratify_real_pred(dur_tensor,indices_evt,indices_cens,q1,q2,X_syn,pred_real,step_rng):

    idx_real_early = indices_evt[(dur_tensor[indices_evt] <= q1)]
    idx_real_mid = indices_evt[(dur_tensor[indices_evt] > q1) & (dur_tensor[indices_evt] <= q2)]
    idx_real_late = indices_evt[(dur_tensor[indices_evt] > q2)]

    real_strata = [idx_real_early, idx_real_mid, idx_real_late, indices_cens]
    sampled_real_indices = stratified_oversample(real_strata, X_syn.shape[0], step_rng)

    pred_real_cat = pred_real[sampled_real_indices] 
    return pred_real_cat


# Evaluate Function used during training
def evaluate(X_syn,T_syn,E_syn,dur_val,evt_val,xgb_params,X_val,dval):
    d_syn = xgb.DMatrix(X_syn.detach().numpy())
    d_syn.set_float_info("label_lower_bound", T_syn)
    d_syn.set_float_info("label_upper_bound", np.where(E_syn == 1, T_syn, np.inf))
    model_eval = xgb.train(xgb_params, d_syn, num_boost_round=100, verbose_eval=False)
    model_eval = xgb.train(params=xgb_params,dtrain=d_syn,evals=[(dval, "validation")],num_boost_round=1000,early_stopping_rounds=50,verbose_eval=False)
    val_idx = (~np.isnan(dur_val)) & (~np.isinf(dur_val)) & (evt_val >= 0)
    d_val = xgb.DMatrix(X_val[val_idx])
    risk_scores = model_eval.predict(d_val)
    c_index = concordance_index_censored(evt_val[val_idx] == 1, dur_val[val_idx], -risk_scores)[0]
    return c_index



### Privacy Accounting Class
class SelectiveAccountant:
    def __init__(self, accountant, max_steps):
        self.accountant = accountant
        self.max_steps = max_steps
        self.current_step = 0

    def step(self, noise_multiplier, sample_rate):
        if self.current_step <= self.max_steps:
            self.accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        self.current_step += 1

    def get_privacy_spent(self, delta):
        return self.accountant.get_privacy_spent(delta=delta)



def bootstrap_cindex_ci(durations, events, risks, n_bootstrap=100, alpha=0.05, seed=42):
    """
    Compute bootstrapped confidence interval for concordance index.

    Parameters:
        durations: array-like of survival durations
        events: array-like of event indicators (1=event, 0=censored)
        risks: array-like of predicted risk scores (higher = higher risk)
        n_bootstrap: number of bootstrap samples
        alpha: significance level for CI (default 0.05 = 95% CI)
        seed: random seed for reproducibility

    Returns:
        mean_cindex: float
        lower: lower bound of CI
        upper: upper bound of CI
    """
    assert len(durations) == len(events) == len(risks), "All input arrays must be the same length."
    
    rng = np.random.default_rng(seed)
    n = len(durations)
    cindex_scores = []

    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping C-index"):
        idx = rng.choice(n, n, replace=True)
        cindex = concordance_index(
            durations[idx],
            -risks[idx],  # higher risk → earlier failure
            events[idx]
        )
        cindex_scores.append(cindex)

    cindex_scores = np.array(cindex_scores)
    lower, upper = np.percentile(cindex_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean_cindex = np.mean(cindex_scores)

    return mean_cindex, lower, upper


def coxph_loss(hazards, durations, events):
    """Survival loss for CoxPH outputs with fixed labels"""
    # Sort by descending time
    order = torch.argsort(durations, descending=True)
    hazards = hazards[order]
    events = events[order]
    
    # Log-sum-exp of hazards for risk sets
    log_risk = torch.logcumsumexp(hazards, dim=0)
    
    # Negative partial likelihood
    loss = -torch.sum((hazards - log_risk) * events) / torch.sum(events)
    return loss
