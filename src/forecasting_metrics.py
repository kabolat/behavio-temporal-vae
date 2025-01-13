import numpy as np

def calculate_quantile_loss(targets, quantile_predictions, quantiles):
    quantiles_losses = [np.maximum(q * (targets - quantile_predictions[i]), (q - 1) * (targets - quantile_predictions[i])) for i, q in enumerate(quantiles)]
    total_loss = np.mean(quantiles_losses)
    return total_loss, quantiles_losses

def calculate_coverage_score(targets, quantile_predictions, quantiles):
    coverage = [np.mean(targets <= quantile_predictions[i]) for i in range(len(quantiles))]

    score = np.mean(np.abs(coverage - np.array(quantiles)))

    return score, coverage

def calculate_interval_score(targets, quantile_predictions, quantiles):
    quantiles = np.array(quantiles)
    iqrs, scores = [], []

    upper_quantiles = quantiles[quantiles > 0.5]
    lower_quantiles = quantiles[quantiles < 0.5]

    for i, q_upper in enumerate(upper_quantiles):
        lower_idx = np.where(lower_quantiles+q_upper==1)[0]
        if len(lower_idx) == 0: continue
        penalties_upper = np.maximum(0, targets - quantile_predictions[i])
        penalties_lower = np.maximum(0, quantile_predictions[lower_idx] - targets)
        iqrs.append((q_upper - quantiles[lower_idx][0]))
        scores.append(((quantile_predictions[i] - quantile_predictions[lower_idx]) + (2 / iqrs[-1]) * penalties_lower + (2 / iqrs[-1]) * penalties_upper).mean())
    
    score = np.mean(scores)

    return score, scores, iqrs

def calculate_interval_coverage_score(targets, quantile_predictions, quantiles):
    quantiles = np.array(quantiles)
    iqrs, interval_coverage = [], []

    upper_quantiles = quantiles[quantiles > 0.5]
    lower_quantiles = quantiles[quantiles < 0.5]

    for i, q_upper in enumerate(upper_quantiles):
        lower_idx = np.where(lower_quantiles+q_upper==1)[0]
        if len(lower_idx) == 0: continue
        lower_bound = quantile_predictions[lower_idx]
        upper_bound = quantile_predictions[i]
        iqrs.append((q_upper - quantiles[lower_idx][0]))
        interval_coverage.append(np.mean((targets >= lower_bound) & (targets <= upper_bound)))

    score = np.mean(np.abs(np.array(interval_coverage) - np.array(iqrs)))

    return score, interval_coverage, iqrs

def calculate_mean_absolute_deviation(targets, quantile_predictions, quantiles):
    mean_absolute_deviations = [np.mean(np.abs(targets - quantile_predictions[i])) for i in range(len(quantiles))]

    score = np.mean(mean_absolute_deviations)

    return score, mean_absolute_deviations