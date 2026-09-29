import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-12)
    gamma = np.std(y_pred) / (np.std(y_true) + 1e-12)
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return {
        'RMSE': rmse, 'MAE': mae, 'R2': r2, 'KGE': kge,
        'Correlation': r, 'BiasRatio': beta, 'VarRatio': gamma
    }


def compute_seasonal_metrics_from_arrays(targets, preds, season_months):
    seasonal_list = []
    for i, month in enumerate(season_months):
        mm = compute_metrics(targets[:, i, 0], preds[:, i, 0])
        mm['month'] = month
        mm['season_index'] = i + 1
        seasonal_list.append(mm)
    return seasonal_list


def print_seasonal_metrics_table(seasonal_metrics, season_name, season_months,
                                  title="Seasonal Regression Metrics"):
    print(f"\n{title} - {season_name} (Months: {season_months})")
    print("=" * 80)
    print(f"{'Month':>6} | {'RMSE':>8} | {'MAE':>8} | {'R2':>8} | {'KGE':>8} | "
          f"{'Corr':>8} | {'BiasRatio':>10} | {'VarRatio':>10}")
    print("-" * 80)
    for mm in seasonal_metrics:
        print(f"{mm['month']:>6} | {mm['RMSE']:>8.4f} | {mm['MAE']:>8.4f} | {mm['R2']:>8.4f} | "
              f"{mm['KGE']:>8.4f} | {mm['Correlation']:>8.4f} | "
              f"{mm['BiasRatio']:>10.4f} | {mm['VarRatio']:>10.4f}")
    print("=" * 80)


def plot_seasonal_results(history, y_true, y_pred, metrics, seasonal_metrics=None,
                          season_name=None, season_months=None, save_path='results.png'):
    season_months = list(season_months) if season_months is not None else None
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    y_t = y_true.flatten()
    y_p = y_pred.flatten()
    residuals = y_p - y_t

    ax = axes[0, 0]
    ax.hexbin(y_t, y_p, gridsize=50, cmap='Blues', mincnt=1)
    ax.plot([y_t.min(), y_t.max()], [y_t.min(), y_t.max()], 'r--', lw=1.5, label='1:1')
    z = np.polyfit(y_t, y_p, 1)
    ax.plot(np.sort(y_t), np.poly1d(z)(np.sort(y_t)), 'k-', lw=1.2,
            label=f'y={z[0]:.2f}x+{z[1]:.2f}')
    ax.set_xlabel('Observed NDVI')
    ax.set_ylabel('Predicted NDVI')
    ax.set_title(f'{season_name} Scatter\nR2={metrics["R2"]:.3f}, '
                 f'RMSE={metrics["RMSE"]:.4f}, KGE={metrics["KGE"]:.3f}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8, density=True)
    ax.axvline(x=0, color='r', linestyle='--', lw=1.5)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Density')
    ax.set_title(f'Residuals\nBias={np.mean(residuals):.4f}, Std={np.std(residuals):.4f}')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    if seasonal_metrics is not None and season_months is not None:
        months = [m['month'] for m in seasonal_metrics]
        r2_list = [m['R2'] for m in seasonal_metrics]
        rmse_list = [m['RMSE'] for m in seasonal_metrics]
        x = np.arange(len(months))
        width = 0.35
        ax.bar(x - width / 2, r2_list, width, label='R2', color='royalblue')
        ax2 = ax.twinx()
        ax2.bar(x + width / 2, rmse_list, width, label='RMSE', color='tomato')
        ax.set_xlabel('Month')
        ax.set_ylabel('R2')
        ax2.set_ylabel('RMSE')
        ax.set_title(f'{season_name} Monthly R2 / RMSE')
        ax.set_xticks(x)
        ax.set_xticklabels(months)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No seasonal metrics', ha='center', va='center',
                transform=ax.transAxes)

    ax = axes[1, 0]
    np.random.seed(123)
    n_show = min(3, len(y_true))
    idxs = np.random.choice(len(y_true), n_show, replace=False)
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    for i, idx in enumerate(idxs):
        ax.plot(season_months, y_true[idx], 'o-', color=colors[i], lw=2, label=f'Obs {idx}')
        ax.plot(season_months, y_pred[idx], 's--', color=colors[i], lw=1.5,
                markerfacecolor='none', label=f'Pred {idx}')
    ax.set_xlabel('Month')
    ax.set_ylabel('NDVI')
    ax.set_title(f'{season_name} Time Series')
    ax.set_xticks(season_months)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(history['train_loss'], label='Train MSE', lw=2)
    ax.plot(history['val_loss'], label='Val MSE', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1, 2]
    ax.axis('off')
    textstr = f"Season: {season_name}\nMonths: {season_months}\n\n"
    textstr += '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    ax.text(0.1, 0.5, textstr, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")