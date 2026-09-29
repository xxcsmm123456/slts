import numpy as np
import os
import rasterio
from rasterio.windows import Window
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch.utils.data import Dataset


DYNAMIC_DRIVERS = ['GPP', 'LST', 'NPP', 'Precipitation', 'Soil_Moisture', 'VPD', 'Radiation']
RENAME_MAP = {
    'GPP': 'GPP', 'LST': 'LST', 'NPP': 'NPP',
    'Precipitation': 'Prec', 'Soil_Moisture': 'SM',
    'VPD': 'VPD', 'Radiation': 'Rad'
}


def read_single_band(path, window, band=1, nodata=np.nan):
    with rasterio.open(path) as src:
        arr = src.read(band, window=window).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = nodata
        return arr


def read_multiband(path, window, bands=None, nodata=np.nan):
    with rasterio.open(path) as src:
        if bands is None:
            bands = list(range(1, src.count + 1))
        arr = src.read(bands, window=window).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = nodata
        return arr


def _format_lc_path(lc_template, year):
    if '{year}' in lc_template:
        return lc_template.format(year=year)
    return lc_template


def load_seasonal_data(driver_template, ndvi_template, static_drivers, lc_template,
                       season, season_months, years, cache_dir=None,
                       max_workers=8, force_reload=False):
    if cache_dir is None:
        cache_dir = os.path.dirname(driver_template)
    os.makedirs(cache_dir, exist_ok=True)

    is_winter = (season == 'Winter')
    if is_winter:
        effective_years = [y for y in years if y not in [2001, 2024]]
        prev_years = [y - 1 for y in effective_years]
        load_years = sorted(set(prev_years + effective_years))
    else:
        effective_years = list(years)
        load_years = list(years)

    cache_file = os.path.join(cache_dir, f'data_cache_{season}_{load_years[0]}_{load_years[-1]}.npz')
    if os.path.exists(cache_file) and not force_reload:
        print(f"Loading cached data from: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        print(f"Loaded: X={data['X'].shape}, y={data['y'].shape}, "
              f"pixels={len(data['pixel_ids'])}, years={list(data['effective_years'])}")
        return (data['X'], data['y'], data['pixel_ids'], data['coords'],
                data['meta'].item(), list(data['effective_years']))

    print(f"Loading data: season={season}, months={season_months}, years={years}")

    first_ndvi = ndvi_template.format(year=load_years[0])
    with rasterio.open(first_ndvi) as src:
        min_h, min_w = src.height, src.width

    for y in load_years:
        for d in DYNAMIC_DRIVERS:
            with rasterio.open(driver_template.format(driver=d, year=y)) as src:
                min_h = min(min_h, src.height)
                min_w = min(min_w, src.width)
    for p in static_drivers.values():
        with rasterio.open(p) as src:
            min_h = min(min_h, src.height)
            min_w = min(min_w, src.width)
    for y in load_years:
        with rasterio.open(_format_lc_path(lc_template, y)) as src:
            min_h = min(min_h, src.height)
            min_w = min(min_w, src.width)

    height, width = min_h, min_w
    print(f"Uniform raster size: {height} x {width}")
    window = Window(0, 0, width, height)

    with rasterio.open(first_ndvi) as src:
        meta = src.meta.copy()
        meta.update({'height': height, 'width': width,
                     'transform': src.window_transform(window), 'nodata': -9999.0})
    transform = meta['transform']

    file_tasks = []
    for name, path in static_drivers.items():
        file_tasks.append((path, 'static', name, None))
    for y in load_years:
        file_tasks.append((_format_lc_path(lc_template, y), 'lc', None, y))
    for y in load_years:
        for d in DYNAMIC_DRIVERS:
            file_tasks.append((driver_template.format(driver=d, year=y), 'dynamic', RENAME_MAP[d], y))
    for y in load_years:
        file_tasks.append((ndvi_template.format(year=y), 'ndvi', None, y))

    results = {}
    print(f"Reading {len(file_tasks)} files with {max_workers} threads")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {}
        for path, ftype, name, year in file_tasks:
            if ftype in ['static', 'lc']:
                future = executor.submit(read_single_band, path, window, 1)
            else:
                future = executor.submit(read_multiband, path, window, None)
            future_to_info[future] = (ftype, name, year)

        for future in as_completed(future_to_info):
            ftype, name, year = future_to_info[future]
            arr = future.result()
            if ftype in ['dynamic', 'ndvi']:
                if arr.shape[0] == 1:
                    arr = np.repeat(arr, 12, axis=0)
                elif arr.shape[0] != 12:
                    raise ValueError(f"{ftype} {name} {year} bands={arr.shape[0]}, expected 12")
            if ftype == 'static':
                results[('static', name)] = arr
            else:
                results[(ftype, year, name)] = arr

    print(f"All files read in {time.time() - t0:.2f}s")

    valid_mask = np.ones((height, width), dtype=bool)
    for name in static_drivers:
        valid_mask &= ~np.isnan(results[('static', name)])
    for y in load_years:
        lc_arr = results[('lc', y, None)]
        valid_mask &= (lc_arr != 0) & ~np.isnan(lc_arr)
    for y in load_years:
        valid_mask &= ~np.any(np.isnan(results[('ndvi', y, None)]), axis=0)
        for d in DYNAMIC_DRIVERS:
            valid_mask &= ~np.any(np.isnan(results[('dynamic', y, RENAME_MAP[d])]), axis=0)

    rows, cols = np.where(valid_mask)
    n_pixels = len(rows)
    print(f"Valid pixels: {n_pixels}")
    if n_pixels == 0:
        raise ValueError("No valid pixels found")

    col_arr = cols.astype(np.float64)
    row_arr = rows.astype(np.float64)
    lon = transform[0] * col_arr + transform[1] * row_arr + transform[2]
    lat = transform[3] * col_arr + transform[4] * row_arr + transform[5]
    coords = np.stack([lon, lat], axis=1)
    pixel_ids = rows * width + cols

    n_features = len(DYNAMIC_DRIVERS) + len(static_drivers)
    n_season_months = len(season_months)
    n_eff_years = len(effective_years)

    X = np.zeros((n_pixels, n_eff_years, n_season_months, n_features), dtype=np.float32)
    y = np.zeros((n_pixels, n_eff_years, n_season_months, 1), dtype=np.float32)

    feature_names = [RENAME_MAP[d] for d in DYNAMIC_DRIVERS] + list(static_drivers.keys())
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    month_to_idx = {m: i for i, m in enumerate(range(1, 13))}

    for eff_idx, eff_year in enumerate(effective_years):
        if is_winter:
            for month_idx, month in enumerate(season_months):
                src_year = eff_year - 1 if month == 12 else eff_year
                for d in DYNAMIC_DRIVERS:
                    short = RENAME_MAP[d]
                    arr = results[('dynamic', src_year, short)]
                    X[:, eff_idx, month_idx, feature_to_idx[short]] = arr[month_to_idx[month], valid_mask]
                ndvi_arr = results[('ndvi', src_year, None)]
                y[:, eff_idx, month_idx, 0] = ndvi_arr[month_to_idx[month], valid_mask]
        else:
            for month_idx, month in enumerate(season_months):
                for d in DYNAMIC_DRIVERS:
                    short = RENAME_MAP[d]
                    arr = results[('dynamic', eff_year, short)]
                    X[:, eff_idx, month_idx, feature_to_idx[short]] = arr[month_to_idx[month], valid_mask]
                ndvi_arr = results[('ndvi', eff_year, None)]
                y[:, eff_idx, month_idx, 0] = ndvi_arr[month_to_idx[month], valid_mask]

        for name in static_drivers:
            vals = results[('static', name)][valid_mask]
            X[:, eff_idx, :, feature_to_idx[name]] = np.repeat(vals[:, np.newaxis], n_season_months, axis=1)

    print(f"Effective years: {effective_years}")
    print(f"Data shape: X={X.shape}, y={y.shape}")

    np.savez_compressed(cache_file, X=X, y=y, pixel_ids=pixel_ids, coords=coords,
                        meta=meta, effective_years=np.array(effective_years))
    print(f"Cache saved: {cache_file}")

    return X, y, pixel_ids, coords, meta, effective_years


def split_data_per_year_then_merge(X, y, effective_years=None,
                                   train_ratio=0.6, val_ratio=0.2, seed=42):
    if effective_years is None:
        effective_years = list(range(X.shape[1]))
    n_pixels, n_years, seq_len, n_features = X.shape
    all_train_idx, all_val_idx, all_test_idx = [], [], []

    for year_idx in range(n_years):
        np.random.seed(seed + year_idx)
        indices = np.random.permutation(n_pixels)
        n_train = int(n_pixels * train_ratio)
        n_val = int(n_pixels * val_ratio)
        all_train_idx.append(indices[:n_train])
        all_val_idx.append(indices[n_train:n_train + n_val])
        all_test_idx.append(indices[n_train + n_val:])
        print(f"Year {effective_years[year_idx]}: train={n_train}, val={n_val}, "
              f"test={n_pixels - n_train - n_val}")

    def extract(pix_idx_list):
        X_list, y_list = [], []
        for year_idx, pix_idx in enumerate(pix_idx_list):
            if len(pix_idx) == 0:
                continue
            X_list.append(X[pix_idx, year_idx, :, :])
            y_list.append(y[pix_idx, year_idx, :, :])
        if not X_list:
            return np.empty((0, seq_len, n_features)), np.empty((0, seq_len, 1))
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    X_train, y_train = extract(all_train_idx)
    X_val, y_val = extract(all_val_idx)
    X_test, y_test = extract(all_test_idx)
    print(f"Merged: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


class NDVIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]