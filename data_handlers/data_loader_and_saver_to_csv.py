import wfdb
from wfdb import processing
import numpy as np
import csv

from data_handlers.DataLoaderConfigBuilder import DataLoaderConfigBuilder

"""
    Process a single ECG signal to extract heartbeat segments.

    Parameters:
    - signal_index (int): Index of the ECG signal to be processed.
    - config (dict): Configuration dictionary containing parameters.

    Returns:
    - segments (list): List of tuples, each containing a heartbeat segment and its associated signal index.
"""


def process_signal(signal_index, config):
    signal_loc = config['recs_folder_path'] + str(signal_index)
    record = wfdb.rdrecord(signal_loc, channels=[0])
    annotation = wfdb.rdann(signal_loc, 'atr')

    qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal, fs=record.fs)

    segments = []
    for r_peak in qrs_inds:
        start_index = max(0, r_peak - config['window_size_before'])
        end_index = min(len(record.p_signal), r_peak + config['window_size_after'])

        if end_index - start_index == config['window_size_before'] + config['window_size_after']:
            beat_annotation_index = np.where((annotation.sample >= start_index) &
                                             (annotation.sample < end_index))[0]

            if beat_annotation_index.size > 0:
                heartbeat_segment = record.p_signal[start_index - 1:end_index]
                label = 0.0
                if annotation.symbol[beat_annotation_index[0]] == 'N':
                    label = 0.0
                elif annotation.symbol[beat_annotation_index[0]] == 'V':
                    label = 1.0
                else:
                    print(
                        f"Skipping heartbeat at index {annotation.sample[beat_annotation_index[0]]} due to unneeded "
                        f"label: "
                        f"{annotation.symbol[beat_annotation_index[0]]}\n")
                    continue

                heartbeat_segment = np.insert(heartbeat_segment, 0, label)
                if np.any(np.isnan(heartbeat_segment)):
                    continue
                segments.append((heartbeat_segment, signal_index))
            else:
                print(f"No annotations in range {start_index} to {end_index}\n")
                continue
        else:
            print(f"Error: Incomplete segmentation from {start_index} to {end_index}\n")

    return segments


"""
    Save heartbeat segments to a CSV file.

    Parameters:
    - segments (list): List of tuples, each containing a heartbeat segment and its associated signal index.
    - csv_file_name_prefix (str): Prefix for the CSV file name.
"""


def save_to_csv(segments, csv_file_name_prefix):
    with open(f"{csv_file_name_prefix}.csv", 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for segment in segments:
            heartbeat_segment, signal_index = segment
            heartbeat_segment = np.insert(heartbeat_segment, 1, signal_index)
            csv_writer.writerow(heartbeat_segment)


"""
    Load ECG data and save heartbeat segments to CSV files.

    Parameters:
    - total_sample_number (int): Number of ECG signals to process.
    - signal_start_index (int): Starting index for ECG signals.
    - config (dict): Configuration dictionary containing parameters.
"""


def data_load_and_save_to_csv(total_sample_number=1, signal_start_index=100, config=None):
    if config is None:
        config = DataLoaderConfigBuilder().build()

    for i in range(signal_start_index, signal_start_index + total_sample_number):
        if i in config['missing_recs']:
            print(f"RECORD #{i} does not exist\n")
            continue
        else:
            print(f"Loading RECORD #{i}\n")
        segments = process_signal(i, config)

        if segments:
            csv_file_name_prefix = config['save_file_path'] + "/" + str(i) + "_heartbeats"
            save_to_csv(segments, csv_file_name_prefix)

    print("Done with loading and saving!\n")


custom_config = DataLoaderConfigBuilder().set_recs_folder_path("../original_signals/") \
    .set_save_file_path("../ecg_csv_files/") \
    .set_window_size_before(99) \
    .set_window_size_after(200) \
    .set_missing_recs([110, 120, 204, 206, 211, 216, 218, 224, 225, 226, 227, 229]) \
    .build()

""" 
    uncomment and put the correct #of records and start index to load the data from wfdb
    if start index is 100, then number of total_sample_number is 25
    if start index is 200, then number of total_sample_number is 35
"""
# data_load_and_save_to_csv(total_sample_number=35, signal_start_index=200, config=custom_config)
