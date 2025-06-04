import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    auc
)
from typing import Tuple, Dict, Union, List

class BinaryClassifierEvaluator:
    def __init__(self, rng_seed: int = 25):
        self.rng = np.random.RandomState(rng_seed)
        self.metric_names = ['Recall', 'Precision', 'F1-Score', 'Specificity', 'PPV', 'NPV', 'AUC']

    def _calculate_prevalence(self, y_true: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        unique, counts = np.unique(y_true, return_counts=True)
        if len(unique) == 1:
            return 0.0 if unique[0] == 1 else 1.0
        return counts[1] / len(y_true)

    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        return numerator / denominator if denominator != 0 else default

    def get_threshold(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        metric_of_interest: str = 'Recall',
        desired_metric_value: float = 0.85,
        error_margin: float = 0.05,
        n_thresholds: int = 1000
    ) -> Tuple[float, pd.DataFrame]:
        y_prob = np.asarray(y_prob)
        y_true = np.asarray(y_true)

        if len(y_prob) == 0 or len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        if y_prob.shape != y_true.shape:
            raise ValueError("y_prob and y_true must have same shape")
        if len(np.unique(y_true)) < 2:
            return 0.5, pd.DataFrame()

        thresholds = np.linspace(0, 1, n_thresholds)
        results = pd.DataFrame(index=thresholds, columns=self.metric_names)
        prev = self._calculate_prevalence(y_true)

        try:
            auc_value = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_value = 0.5

        for t in thresholds:
            pred = (y_prob > t).astype(int)
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            except ValueError:
                tn, fp, fn, tp = 0, 0, 0, 0

            recall = self._safe_divide(tp, tp + fn)
            precision = self._safe_divide(tp, tp + fp, default=1.0)
            specificity = self._safe_divide(tn, tn + fp, default=1.0)

            ppv_num = recall * prev
            ppv_denom = ppv_num + (1 - specificity) * (1 - prev)
            ppv = self._safe_divide(ppv_num, ppv_denom)

            npv_num = specificity * (1 - prev)
            npv_denom = npv_num + (1 - recall) * prev
            npv = self._safe_divide(npv_num, npv_denom, default=1.0)

            f1 = self._safe_divide(2 * precision * recall, (precision + recall))

            results.loc[t] = [recall, precision, f1, specificity, ppv, npv, auc_value]

        target_metric = results[metric_of_interest]
        mask = (target_metric >= desired_metric_value - error_margin) & (target_metric <= desired_metric_value + error_margin)

        if not mask.any():
            closest_idx = float((target_metric - desired_metric_value).abs().idxmin())
            print(f"No threshold found within margin. Returning closest: {closest_idx:.3f}")
            try:
                closest_row = results.loc[[closest_idx]]
            except KeyError:
                closest_row = results.iloc[[results.index.get_loc(closest_idx)]]
            return closest_idx, closest_row

        sort_metric = {
            'Recall': 'Precision',
            'Precision': 'Recall',
            'Specificity': 'Recall',
        }.get(metric_of_interest, metric_of_interest)

        best_threshold = results[mask].sort_values(sort_metric, ascending=False).index[0]
        return best_threshold, results.loc[[best_threshold]]

    def _bootstrap_metric(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric_func: callable,
        n_bootstraps: int = 1000,
        **metric_kwargs
    ) -> Tuple[float, float]:
        scores = []
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        idx_0 = np.where(y_true == 0)[0]
        idx_1 = np.where(y_true == 1)[0]
        n_0, n_1 = len(idx_0), len(idx_1)

        if n_0 == 0 or n_1 == 0:
            return (0.0, 0.0)

        for _ in range(n_bootstraps):
            resampled_idx_0 = self.rng.choice(idx_0, n_0, replace=True)
            resampled_idx_1 = self.rng.choice(idx_1, n_1, replace=True)
            indices = np.concatenate([resampled_idx_0, resampled_idx_1])
            try:
                score = metric_func(y_true[indices], y_prob[indices], **metric_kwargs)
                scores.append(score)
            except:
                continue

        if not scores:
            return (0.0, 0.0)

        sorted_scores = np.sort(scores)
        lower_idx = int(0.025 * len(sorted_scores))
        upper_idx = int(0.975 * len(sorted_scores))
        return (sorted_scores[lower_idx], sorted_scores[upper_idx])

    def get_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
        n_bootstraps: int = 1000
    ) -> Dict[str, Tuple[float, float]]:
        if len(y_true) == 0 or len(y_prob) == 0:
            return {name: (0.0, 0.0) for name in ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'PPV', 'NPV']}

        y_pred = (y_prob > threshold).astype(int)
        prev = self._calculate_prevalence(y_true)

        def ppv_score(y_true, y_pred, prev):
            try:
                sens = recall_score(y_true, y_pred)
                spec = recall_score(y_true, y_pred, pos_label=0)
                return self._safe_divide(sens * prev, sens * prev + (1 - spec) * (1 - prev))
            except:
                return 0.0

        def npv_score(y_true, y_pred, prev):
            try:
                sens = recall_score(y_true, y_pred)
                spec = recall_score(y_true, y_pred, pos_label=0)
                return self._safe_divide(spec * (1 - prev), spec * (1 - prev) + (1 - sens) * prev)
            except:
                return 0.0

        metrics = {
            'AUROC': (roc_auc_score, {}),
            'AUPRC': (average_precision_score, {}),
            'Sensitivity': (recall_score, {}),
            'Specificity': (lambda y, p: recall_score(y, p, pos_label=0), {}),
            'PPV': (ppv_score, {'prev': prev}),
            'NPV': (npv_score, {'prev': prev})
        }

        ci_results = {}
        for name, (func, kwargs) in metrics.items():
            ci_results[name] = self._bootstrap_metric(
                y_true,
                y_prob if 'AUC' in name else y_pred,
                func,
                n_bootstraps,
                **kwargs
            )
        return ci_results

    def evaluate_separately(
        self,
        y_val_true: np.ndarray,
        y_val_prob: np.ndarray,
        y_test_true: np.ndarray,
        y_test_prob: np.ndarray,
        target_metric: str = 'Recall',
        target_value: float = 0.85,
        error_margin: float = 0.05,
        n_bootstraps: int = 1000,
        return_ci: bool = False
    ) -> Dict[str, Union[float, str]]:
        for arr in [y_val_true, y_val_prob, y_test_true, y_test_prob]:
            if len(arr) == 0:
                raise ValueError("Input arrays cannot be empty")

        threshold, _ = self.get_threshold(
            y_prob=y_val_prob,
            y_true=y_val_true,
            metric_of_interest=target_metric,
            desired_metric_value=target_value,
            error_margin=error_margin,
            n_thresholds=1000
        )
        print(f"Optimal threshold determined from validation set: {threshold:.3f}")

        y_test_pred = (y_test_prob > threshold).astype(int)
        prev = self._calculate_prevalence(y_test_true)

        metrics = {}

        try:
            metrics['AUROC'] = roc_auc_score(y_test_true, y_test_prob)
        except:
            metrics['AUROC'] = 0.5

        try:
            precision, recall, _ = precision_recall_curve(y_test_true, y_test_prob)
            svc_pr_auc = auc(recall, precision)
            metrics['AUPRC'] = svc_pr_auc
        except:
            metrics['AUPRC'] = prev

        metrics['Sensitivity'] = recall_score(y_test_true, y_test_pred)
        metrics['Specificity'] = recall_score(y_test_true, y_test_pred, pos_label=0)

        metrics['PPV'] = self._safe_divide(
            metrics['Sensitivity'] * prev,
            metrics['Sensitivity'] * prev + (1 - metrics['Specificity']) * (1 - prev)
        )
        metrics['NPV'] = self._safe_divide(
            metrics['Specificity'] * (1 - prev),
            metrics['Specificity'] * (1 - prev) + (1 - metrics['Sensitivity']) * prev
        )

        if return_ci:
            ci = self.get_confidence_intervals(y_test_true, y_test_prob, threshold, n_bootstraps)
            for name, bounds in ci.items():
                if name in metrics:
                    center = metrics[name]
                    lower, upper = bounds
                    half_width = (upper - lower) / 2
                    metrics[name] = f"{center:.3f} ± {half_width:.3f}"

        return metrics
