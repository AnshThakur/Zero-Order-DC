import torch
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from sksurv.metrics import concordance_index_censored
from opacus.accountants import RDPAccountant


def stratified_oversample(real_indices_by_stratum, total_target, generator):
    """Performs stratified oversampling from input index groups to match target size."""
    all_real_indices = []
    weights = []
    for group in real_indices_by_stratum:
        weights.append(len(group))
        all_real_indices.append(group)

    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / (weights.sum() + 1e-9)
    group_sizes = torch.round(weights * total_target).int()
    group_sizes[-1] = total_target - group_sizes[:-1].sum()

    sampled_indices = []
    for group, size in zip(all_real_indices, group_sizes):
        if len(group) == 0:
            continue
        group_tensor = group.clone() if isinstance(group, torch.Tensor) else torch.tensor(group)
        selected = group_tensor[torch.randint(0, len(group), (size,), generator=generator)]
        sampled_indices.append(selected)

    return torch.cat(sampled_indices)


class SelectiveAccountant:
    """Tracks DP accounting only up to a maximum number of steps."""
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


def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Computes numerically stable binary cross-entropy."""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def stratify_pred_binary(y_tensor, pred_tensor):
    """Stratifies binary predictions by class."""
    idx_class0 = (y_tensor == 0).nonzero(as_tuple=True)[0]
    idx_class1 = (y_tensor == 1).nonzero(as_tuple=True)[0]
    pred_groups = [pred_tensor[idx_class0], pred_tensor[idx_class1]]
    pred_cat = torch.cat(pred_groups, dim=0)
    return pred_cat, [idx_class0, idx_class1]


def stratify_real_pred_binary(pred_real, idx_groups, X_syn, step_rng):
    """Samples stratified real binary predictions to match synthetic set."""
    sampled_real_indices = stratified_oversample(idx_groups, X_syn.shape[0], step_rng)
    sampled_real_indices = sampled_real_indices.to(pred_real.device)
    return pred_real[sampled_real_indices]


def evaluate(X_syn, y_syn, xgb_params, dval, x_val, y_val):
    """Trains XGBoost on synthetic data and evaluates AUROC."""
    d_syn = xgb.DMatrix(data=X_syn.detach().numpy(), label=y_syn.detach().numpy())
    model_eval = xgb.train(params=xgb_params, dtrain=d_syn, evals=[(dval, "validation")],
                           num_boost_round=1000, early_stopping_rounds=50, verbose_eval=False)
    risk_scores = model_eval.predict(dval)
    auroc = roc_auc_score(y_val, risk_scores)
    return auroc


def stratified_oversample_proto(real_indices_by_stratum, total_target, generator):
    """GPU-compatible stratified oversampling function for prototyping."""
    all_real_indices = []
    weights = []

    for group in real_indices_by_stratum:
        group_tensor = torch.tensor(group, device='cuda') if not isinstance(group, torch.Tensor) else group.to('cuda')
        all_real_indices.append(group_tensor)
        weights.append(len(group_tensor))

    weights = torch.tensor(weights, dtype=torch.float, device='cuda')
    weights = weights / (weights.sum() + 1e-9)

    group_sizes = torch.round(weights * total_target).int()
    group_sizes[-1] = total_target - group_sizes[:-1].sum()

    sampled_indices = []
    for group, size in zip(all_real_indices, group_sizes):
        if len(group) == 0 or size == 0:
            continue
        rand_idx = torch.randint(low=0, high=len(group), size=(size,), generator=generator, device=group.device)
        sampled_indices.append(group[rand_idx])

    return torch.cat(sampled_indices)
