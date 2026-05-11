import os
import numpy as np
import pandas as pd
from natsort import natsorted
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from scipy.fft import fft
from scipy.signal import detrend
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def dir_list(path: str) -> List:
    """
    :param path: path to the dataset
    :return: an order directory list
    """

    list_dir = natsorted(os.listdir(path))
    return list_dir


def get_column_names(num_channels: int):
    """
    :param num_channels: int number of channels in one IMS file
    :return: list with channel names
    """
    if num_channels == 8:
        col_names = []
        for b in range(0, 4):
            b1 = f'b{b + 1}_ch{b * 2 + 1}'
            b2 = f'b{b + 1}_ch{b * 2 + 2}'
            col_names.extend([b1, b2])
        return col_names

    return [f'b{i + 1}_ch{i + 1}' for i in range(0, num_channels)]


def compute_signal_features(signal_values, detrends=True):
    """
    :param signal_values: numpy array with one channel from one IMS file
    :param detrends: bool detrend signal before computing features
    :return: dict with threshold-ready features
    """
    signal = np.asarray(signal_values, dtype=float)
    work_signal = detrend(signal) if detrends else signal

    abs_signal = np.abs(work_signal)
    rms = np.sqrt(np.mean(work_signal ** 2))
    std = np.std(work_signal)
    mean_abs = np.mean(abs_signal)
    peak = np.max(abs_signal)
    peak_to_peak = np.ptp(work_signal)
    crest_factor = peak / rms if rms != 0 else 0.0

    centered = work_signal - np.mean(work_signal)
    centered_std = np.std(centered)
    if centered_std == 0:
        kurtosis = 0.0
    else:
        kurtosis = np.mean((centered / centered_std) ** 4)

    fft_values = np.abs(fft(work_signal)) / len(work_signal)
    spectral_energy = np.sum(fft_values ** 2)

    return {
        'mean_abs': mean_abs,
        'rms': rms,
        'std': std,
        'peak': peak,
        'peak_to_peak': peak_to_peak,
        'crest_factor': crest_factor,
        'kurtosis': kurtosis,
        'spectral_energy': spectral_energy,
    }


def concat_raw_data(path=None, csv_path=None, dataset=None, fourier_tr=True,
                    detrends=True):
    """
    :param detrends: boolean to detrend the signal
    :param fourier_tr: boolean transform signal with Fourier transform
    :param path: a path to the dataset file
    :param dataset: number of the dataset to be processes from the three datasets available
    :param csv_path: path save the concatenated files into a csv file
    :return: dataset with average and std from each file at each time step
    """

    list_dir = dir_list(path)

    dataset_dict = {}

    for i, f in enumerate(list_dir):
        temp_df = pd.read_csv(os.path.join(path, f), sep='\t', header=None)
        temp_df.columns = get_column_names(len(temp_df.columns))
        temp_df.insert(0, 'date', len(temp_df) * [f])
        dataset_dict[f] = temp_df

    df = pd.concat(list(dataset_dict.values()), ignore_index=True)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y.%m.%d.%H.%M.%S')

    os.makedirs(csv_path, exist_ok=True)

    if fourier_tr:
        fourier_transforms(df, path=csv_path, dataset=dataset, detrends=detrends)
        return df

    else:
        fname = os.path.join(csv_path, f'concat_dataset_{dataset}.csv')
        df.to_csv(fname)
        return df


def fourier_transforms(data_frame, path=None, dataset=1, detrends=True):
    """
    :param data_frame: datafram with the concatenated raw data
    :param path: a path to store the csv file
    :param dataset: number of dataset processed
    :param detrends: a boolean to detrend before applying fourier transformations
    :return: save dataframe as csv
    """

    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, f'fft_dataset_{dataset}.csv')
    df_fft = data_frame.copy()

    for col in df_fft.columns:
        if detrends:
            fft_col = fft(detrend(df_fft[col].values))
        else:
            fft_col = fft(df_fft[col].values)
        N = len(fft_col)
        df_fft[col] = np.abs(fft_col) / N

    df_fft.to_csv(fname)
    return df_fft


def average_signal_dataset(path=str , dataset=1,
                           csv_path: str=None, abs_mean: bool = True):
    """
    :param path:  str a path to the dataset file
    :param dataset:  int the number of the dataset to be process
    :param csv_path:  str a path to save the dataframes to csv file
    :return: data dataset with average and std from each file at each time step
    """
    list_dir = dir_list(path)

    if len(list_dir) == 0:
        raise FileNotFoundError(f"No files found in {path}")

    dataset_dict = {}

    for file in list_dir:
        temp_df = pd.read_csv(os.path.join(path, file), sep='\t', header=None)
        temp_df.columns = get_column_names(len(temp_df.columns))
        mean = temp_df.abs().mean().values if abs_mean else temp_df.mean().values
        dataset_dict[file] = mean

    # build dataframe
    first_file = pd.read_csv(os.path.join(path, list_dir[0]), sep='\t', header=None)
    col_names = get_column_names(len(first_file.columns))
    df = pd.DataFrame.from_dict(dataset_dict, orient='index', columns=col_names)

    # parse filenames as datetimes; if it fails, keep string index
    try:
        df.index = pd.to_datetime(df.index, format='%Y.%m.%d.%H.%M.%S')
    except Exception:
        # fallback: keep original string index
        logger.warning("Datetime parsing failed for filenames; keeping string index.")

    df.sort_index(inplace=True)

    # Save to csv
    if csv_path is not None:
        out_dir = Path(csv_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f'avg_concat_dataset_{dataset}.csv'
        df.to_csv(fname)

    return df


def signal_features_dataset(path=None, dataset=1, csv_path=None, detrends=True):
    """
    :param path: str a path to the dataset file
    :param dataset: int the number of the dataset to be processed
    :param csv_path: str a path to save the dataframes to csv file
    :param detrends: bool detrend each file before feature extraction
    :return: dataframe with one row per file and multiple features per channel
    """
    list_dir = dir_list(path)
    feature_rows = {}

    for file in list_dir:
        temp_df = pd.read_csv(os.path.join(path, file), sep='\t', header=None)
        temp_df.columns = get_column_names(len(temp_df.columns))

        row_features = {}
        for col in temp_df.columns:
            feature_dict = compute_signal_features(temp_df[col].values, detrends=detrends)
            for feature_name, feature_value in feature_dict.items():
                row_features[f'{col}_{feature_name}'] = feature_value

        feature_rows[file] = row_features

    df = pd.DataFrame.from_dict(feature_rows, orient='index')
    df.index = pd.to_datetime(df.index, format='%Y.%m.%d.%H.%M.%S')
    os.makedirs(csv_path, exist_ok=True)

    fname = os.path.join(csv_path, f'features_dataset_{dataset}.csv')
    df.to_csv(fname)
    return df


def reshape_features_by_bearing(features_df):
    """
    :param features_df: dataframe with one row per timestamp and one column per chanel-feature
    :return: long dataframe with one row per timestamp per bearing channel
    """
    feature_suffixes = (
        'mean_abs',
        'rms',
        'std',
        'peak',
        'peak_to_peak',
        'crest_factor',
        'kurtosis',
        'spectral_energy',
    )

    long_rows = []
    for timestamp, row in features_df.iterrows():
        bearing_map = {}
        for col_name, value in row.items():
            matched_suffix = None
            for suffix in feature_suffixes:
                token = f'_{suffix}'
                if col_name.endswith(token):
                    matched_suffix = suffix
                    channel_name = col_name[:-len(token)]
                    break

            if matched_suffix is None:
                continue

            bearing_map.setdefault(channel_name, {})[matched_suffix] = value

        for channel_name, feature_dict in bearing_map.items():
            row_dict = {
                'timestamp': timestamp,
                'bearing_id': channel_name,
            }
            row_dict.update(feature_dict)
            long_rows.append(row_dict)

    long_df = pd.DataFrame(long_rows)
    long_df.sort_values(['timestamp', 'bearing_id'], inplace=True)
    long_df.reset_index(drop=True, inplace=True)
    return long_df


def reshape_features_csv(features_csv_path, output_path=None):
    """
    :param features_csv_path: path to wide features csv
    :param output_path: optional output path for reshaped csv
    :return: long dataframe with one row per timestamp per bearing channel
    """
    features_df = pd.read_csv(features_csv_path, index_col=0, parse_dates=True)
    long_df = reshape_features_by_bearing(features_df)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        long_df.to_csv(output_path, index=False)

    return long_df


def generate_datasets(datasets_dict=None, csv_path=None):
    for i, (k, v) in enumerate(datasets_dict.items()):
        dataset_num = i + 1
        average_signal_dataset(v, dataset=dataset_num, csv_path=csv_path)
        signal_features_dataset(v, dataset=dataset_num, csv_path=csv_path, detrends=True)
        concat_raw_data(v, csv_path, dataset=dataset_num, fourier_tr=True, detrends=False)


if __name__ == '__main__':
    print(os.getcwd())
    datasets = {'dataset_path1': './ims_bearing/1st_test/1st_test',
                'dataset_path2': './ims_bearing/2nd_test/2nd_test',
                'dataset_path3': './ims_bearing/3rd_test/4th_test/txt'}

    csv_dir = os.path.join(os.getcwd(), 'csv_data')
    os.makedirs(csv_dir, exist_ok=True)

    generate_datasets(datasets, csv_path=csv_dir)
