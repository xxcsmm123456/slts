import numpy as np
import torch
import os
import pickle
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_seasonal_data, split_data_per_year_then_merge, NDVIDataset
from models import MODEL_REGISTRY
from utils import set_seed, compute_metrics


def predict_model(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc='Predicting', unit='batch'):
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


DEFAULT_MODEL_PARAMS = {
    'LSTMTransformer': {
        'd_model': 64, 'hidden_size': 128, 'num_layers': 2,
        'num_transformer_layers': 3, 'nhead': 4, 'dim_feedforward': 256, 'dropout': 0.1,
    },
}


def predict_pipeline(model_name, season, season_months, years,
                     driver_template, ndvi_template, static_drivers, lc_template,
                     model_path, scaler_path, output_dir, cache_dir=None,
                     batch_size=2048, train_ratio=0.6, val_ratio=0.2,
                     force_reload=False):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'=' * 70}")
    print(f"Predicting: {model_name} - {season}")
    print(f"Model: {model_path}")
    print(f"{'=' * 70}")

    X, y, pixel_ids, coords, meta, effective_years = load_seasonal_data(
        driver_template, ndvi_template, static_drivers, lc_template,
        season, season_months, years, cache_dir=cache_dir, force_reload=force_reload)

    n_pixels, n_years, seq_len, n_features = X.shape
    print(f"Data: X={X.shape}, y={y.shape}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_data_per_year_then_merge(
        X, y, effective_years, train_ratio, val_ratio)

    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']

    X_test_scaled = scaler_X.transform(
        X_test.reshape(-1, n_features)).reshape(X_test.shape)
    y_test_scaled = scaler_y.transform(
        y_test.reshape(-1, 1)).reshape(y_test.shape)

    test_loader = DataLoader(NDVIDataset(X_test_scaled, y_test_scaled),
                             batch_size=batch_size, shuffle=False)

    model_class = MODEL_REGISTRY[model_name]
    params = dict(DEFAULT_MODEL_PARAMS.get(model_name, {}))
    params['input_features'] = n_features
    params['seq_len'] = seq_len
    model = model_class(**params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    preds_scaled, targets_scaled = predict_model(model, test_loader, device)
    preds = scaler_y.inverse_transform(
        preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)
    targets = scaler_y.inverse_transform(
        targets_scaled.reshape(-1, 1)).reshape(targets_scaled.shape)

    metrics = compute_metrics(targets, preds)
    print(f"\n{model_name} - {season} Metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    out_path = os.path.join(output_dir, f'{model_name.lower()}_{season.lower()}_predictions.npz')
    np.savez_compressed(out_path, preds=preds, targets=targets,
                        effective_years=np.array(effective_years))
    print(f"Predictions saved: {out_path}")

    return metrics, preds, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTMTransformer',
                        choices=['LSTMTransformer'])
    parser.add_argument('--season', type=str, default='Year')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--scaler-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--data-root', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=2048)
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

    predict_pipeline(
        model_name=args.model,
        season=args.season,
        season_months=season_months,
        years=raw_years,
        driver_template=driver_template,
        ndvi_template=ndvi_template,
        static_drivers=static_drivers,
        lc_template=lc_template,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        force_reload=args.force_reload
    )


if __name__ == '__main__':
    main()