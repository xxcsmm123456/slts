import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
import pickle
from tqdm import tqdm
import pandas as pd
import argparse

from data import load_seasonal_data, split_data_per_year_then_merge, NDVIDataset
from models import MODEL_REGISTRY
from utils import (set_seed, compute_metrics, compute_seasonal_metrics_from_arrays,
                   print_seasonal_metrics_table, plot_seasonal_results)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs}', unit='batch')
    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        avg = total_loss / ((batch_idx + 1) * dataloader.batch_size)
        pbar.set_postfix({'batch_loss': f'{loss.item():.4f}', 'avg_loss': f'{avg:.4f}'})
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    r2 = r2_score(targets.flatten(), preds.flatten())
    return avg_loss, preds, targets, r2


def train_model(model, train_loader, val_loader, test_loader=None, eval_every=5,
                scaler_y=None, season_name=None, season_months=None,
                epochs=100, device='cuda', lr=1e-3, weight_decay=1e-5, patience=15,
                save_dir='checkpoints', save_best_only=False):
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stop = EarlyStopping(patience=patience, min_delta=1e-5)

    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'test_seasonal_metrics': []}
    best_model_state = None
    best_val_loss = float('inf')

    torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_epoch_0.pth'))

    print(f"\n{'=' * 80}")
    print(f"Training - {season_name} (Months: {season_months})")
    print(f"Checkpoints: {save_dir}")
    print(f"{'=' * 80}")
    print(f"{'Epoch':>6} | {'TrainMSE':>12} | {'ValMSE':>12} | {'ValR2':>10} | "
          f"{'LR':>12} | {'BestValMSE':>12}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, epoch, epochs)
        val_loss, _, _, val_r2 = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        early_stop(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {val_r2:>10.4f} | "
              f"{current_lr:>12.2e} | {best_val_loss:>12.6f}")

        if not save_best_only:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_r2': val_r2,
                'best_val_loss': best_val_loss,
                'season': season_name,
                'season_months': season_months,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

        if test_loader is not None and scaler_y is not None and epoch % eval_every == 0:
            _, preds_scaled, targets_scaled, _ = validate(model, test_loader, criterion, device)
            preds = scaler_y.inverse_transform(
                preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)
            targets = scaler_y.inverse_transform(
                targets_scaled.reshape(-1, 1)).reshape(targets_scaled.shape)
            seasonal_metrics = compute_seasonal_metrics_from_arrays(targets, preds, season_months)
            history['test_seasonal_metrics'].append({'epoch': epoch, 'metrics': seasonal_metrics})
            print(f"\n>>> Epoch {epoch} Test Seasonal ({season_name}) <<<")
            print(f"{'Month':>6} | {'RMSE':>8} | {'R2':>8}")
            print("-" * 28)
            for mm in seasonal_metrics:
                print(f"{mm['month']:>6} | {mm['RMSE']:>8.4f} | {mm['R2']:>8.4f}")
            print()
            pd.DataFrame(seasonal_metrics).to_csv(
                os.path.join(save_dir, f'eval_epoch_{epoch}.csv'), index=False)

        if early_stop.early_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(save_dir, 'best_model.pth'))
        model.load_state_dict(best_model_state)
        print(f"Best model saved: {os.path.join(save_dir, 'best_model.pth')}")

    with open(os.path.join(save_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    print(f"\nTraining complete, best val MSE = {best_val_loss:.6f}")
    return model, history


def train_pipeline(model_name, model_params, season, season_months, years,
                   driver_template, ndvi_template, static_drivers, lc_template,
                   output_dir, cache_dir=None, split_mode='per_year_then_merge',
                   train_ratio=0.6, val_ratio=0.2, batch_size=2048, epochs=150,
                   lr=5e-4, weight_decay=1e-5, patience=20, max_workers=8,
                   eval_every=5, save_best_only=False, force_reload=False):
    os.makedirs(output_dir, exist_ok=True)
    if cache_dir is None:
        cache_dir = output_dir
    os.makedirs(cache_dir, exist_ok=True)

    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'=' * 70}")
    print(f"Model: {model_name}")
    print(f"Season: {season}, Months: {season_months}")
    print(f"Years: {years}")
    print(f"Split mode: {split_mode}")
    print(f"{'=' * 70}")

    X, y, pixel_ids, coords, meta, effective_years = load_seasonal_data(
        driver_template, ndvi_template, static_drivers, lc_template,
        season, season_months, years, cache_dir=cache_dir,
        max_workers=max_workers, force_reload=force_reload)

    n_pixels, n_years, seq_len, n_features = X.shape
    print(f"Data: X={X.shape}, y={y.shape}")

    if split_mode == 'per_year_then_merge':
        X_train, y_train, X_val, y_val, X_test, y_test = split_data_per_year_then_merge(
            X, y, effective_years, train_ratio, val_ratio)
    else:
        X_all = X.reshape(-1, seq_len, n_features)
        y_all = y.reshape(-1, seq_len, 1)
        total = X_all.shape[0]
        np.random.seed(42)
        indices = np.random.permutation(total)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        X_train = X_all[indices[:n_train]]
        y_train = y_all[indices[:n_train]]
        X_val = X_all[indices[n_train:n_train + n_val]]
        y_val = y_all[indices[n_train:n_train + n_val]]
        X_test = X_all[indices[n_train + n_val:]]
        y_test = y_all[indices[n_train + n_val:]]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(
        X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(
        X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(
        X_test.reshape(-1, n_features)).reshape(X_test.shape)

    y_train_scaled = scaler_y.fit_transform(
        y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val_scaled = scaler_y.transform(
        y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test_scaled = scaler_y.transform(
        y_test.reshape(-1, 1)).reshape(y_test.shape)

    train_loader = DataLoader(NDVIDataset(X_train_scaled, y_train_scaled),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NDVIDataset(X_val_scaled, y_val_scaled),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(NDVIDataset(X_test_scaled, y_test_scaled),
                             batch_size=batch_size, shuffle=False)

    model_class = MODEL_REGISTRY[model_name]
    params = dict(model_params)
    params['input_features'] = n_features
    params['seq_len'] = seq_len
    model = model_class(**params).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model, history = train_model(
        model, train_loader, val_loader, test_loader=test_loader,
        eval_every=eval_every, scaler_y=scaler_y, season_name=season,
        season_months=season_months, epochs=epochs, device=device, lr=lr,
        weight_decay=weight_decay, patience=patience, save_dir=checkpoint_dir,
        save_best_only=save_best_only)

    _, preds_scaled, targets_scaled, _ = validate(model, test_loader, nn.MSELoss(), device)
    preds = scaler_y.inverse_transform(
        preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)
    targets = scaler_y.inverse_transform(
        targets_scaled.reshape(-1, 1)).reshape(targets_scaled.shape)

    metrics = compute_metrics(targets, preds)
    print(f"\nTest Metrics - {season}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    seasonal_metrics = compute_seasonal_metrics_from_arrays(targets, preds, season_months)
    print_seasonal_metrics_table(seasonal_metrics, season, season_months,
                                  title="Final Test Seasonal Metrics")

    pd.DataFrame(seasonal_metrics).to_csv(
        os.path.join(output_dir, 'seasonal_metrics_final.csv'), index=False)

    plot_path = os.path.join(output_dir, f'{model_name.lower()}_{season.lower()}_results.png')
    plot_seasonal_results(history, targets, preds, metrics,
                          seasonal_metrics=seasonal_metrics,
                          season_name=season, season_months=season_months,
                          save_path=plot_path)

    torch.save(model.state_dict(),
               os.path.join(output_dir, f'{model_name.lower()}_{season.lower()}_ndvi.pth'))
    with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

    return model, metrics, seasonal_metrics, effective_years


DEFAULT_MODEL_PARAMS = {
    'LSTMTransformer': {
        'd_model': 64, 'hidden_size': 128, 'num_layers': 2,
        'num_transformer_layers': 3, 'nhead': 4, 'dim_feedforward': 256, 'dropout': 0.1,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTMTransformer',
                        choices=['LSTMTransformer'])
    parser.add_argument('--season', type=str, default='Year')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--data-root', type=str, default='/data')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--force-reload', action='store_true')
    args = parser.parse_args()

    set_seed(42)

    driver_template = args.data_root + '/Factor/{driver}/Year_{driver}_{year}_SG.tif'
    ndvi_template = args.data_root + '/Factor/NDVI/Year_NDVI_{year}_SG.tif'
    static_drivers = {
        'DEM': args.data_root + '/Factor/DEM/DEM.tif',
        'Slope': args.data_root + '/Factor/DEM/slop.tif',
        'Aspect': args.data_root + '/Factor/DEM/aspect.tif'
    }
    lc_template = args.data_root + '/Factor/LC/Year_LC_{year}.tif'

    raw_years = list(range(2001, 2025))
    seasons = {
        'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Autumn': [9, 10, 11],
        'Winter': [12, 1, 2], 'Growing': [5, 6, 7, 8, 9],
        'Year': list(range(1, 13))
    }
    season_months = seasons[args.season]

    train_pipeline(
        model_name=args.model,
        model_params=DEFAULT_MODEL_PARAMS.get(args.model, {}),
        season=args.season,
        season_months=season_months,
        years=raw_years,
        driver_template=driver_template,
        ndvi_template=ndvi_template,
        static_drivers=static_drivers,
        lc_template=lc_template,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        force_reload=args.force_reload
    )


if __name__ == '__main__':
    main()