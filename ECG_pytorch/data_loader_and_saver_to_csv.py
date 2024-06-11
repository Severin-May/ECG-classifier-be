"""
@brief: this file loads the data_handlers from wfdb database and saves them as csv files

Database records undergo peak correction and are then segmented based on fixed
R-R intervals using a fixed window size (99 data_handlers samples before and 200 after the R-peak).
Each CSV row represents one heartbeat with information on record name, segment number,
annotation label followed by 300 data_handlers points.
"""

import wfdb
from wfdb import processing
import numpy as np
import csv

"""
@brief: loads records and saves them as csv
@param: total_sample_number - total number of records to be loaded & saved, either 23 or 25
@param: signal_start_index - it is either 100 or 200
@param: window_size_before - # of data points before R-peak
@param: window_size_after - # of data points after R-peak
@param: save_file_path - where to save generated csv file
@return: N/A
# from 100 to 124 -> 23 records, 120, 110 are missing, pass 100 and 24 as params
# check are these two records present on the website
# from 200 to 234 -> 25 records, 204, 206, 211, 216, 218, 224, 225, 226, 227, 229 are missing, pass 200 and 35 as params
"""
missing_recs = [110, 120, 204, 206, 211, 216, 218, 224, 225, 226, 227, 229]


def data_load_and_save_to_csv(total_sample_number=35, signal_start_index=200,
                              window_size_before=99, window_size_after=200,
                              save_file_path="ecg_csv_files/"):
    for i in range(signal_start_index, signal_start_index + total_sample_number):
        # Skipping the missing records
        if i in missing_recs:
            print("record #" + str(i) + " does not exist")
            continue
        else:
            print("record #" + str(i))
            signal_loc = "../original_signals/" + str(i)
            # returns a signal and record descriptors in a Record object
            record = wfdb.rdrecord(signal_loc, channels=[0])
            # returns annotations for a record
            annotation = wfdb.rdann(signal_loc, 'atr')
            # print(record.p_signal[300:380])

        # Using the GQRS algorithm to detect QRS locations in the first channel
        qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal, fs=record.fs)

        # print("qrs_inds: ")
        # print(qrs_inds[:10])

        # Correcting the peaks shifting them to local maxima
        min_bpm = 20
        max_bpm = 230
        search_radius = int(record.fs * 60 / max_bpm)
        corrected_peak_inds = wfdb.processing.correct_peaks(sig=record.p_signal[:, 0], peak_inds=qrs_inds,
                                                            search_radius=search_radius,
                                                            smooth_window_size=150, peak_dir='compare')
        # print("length of corrected_peak_inds: " + str(len(corrected_peak_inds)))
        # print("corrected_peak_inds: ")
        # print(corrected_peak_inds[:10])
        # corrected_peak_inds = corrected_peak_inds[:-1]
        # np.set_printoptions(threshold=np.inf)
        # print(corrected_peak_inds)

        # the below code is just for verification of R-peaks

        # Obtaining R-R interval series from ECG annotation files, return type = numpy.ndarray
        # RR_intervals_ann = wfdb.processing.ann2rr(signal_loc, 'atr', as_array=True)
        # print("length of RR_intervals_ann: " + str(len(RR_intervals_ann)))
        # print("RR_intervals_ann: ")
        # print(RR_intervals_ann[:10])

        # Calculating R-R intervals from qrs indices
        # RR_intervals = wfdb.processing.calc_rr(qrs_inds, fs=None, min_rr=None, max_rr=None,
        #                                        qrs_units='samples', rr_units='samples')
        # print("length of RR_intervals: " + str(len(RR_intervals)))
        # print("RR_intervals: ")
        # print(RR_intervals[:10])

        csv_file_name_prefix = save_file_path + "/" + str(i) + "_heartbeats"

        # Segmentation of ecg signals into heartbeats, corrected_peak_indices store the locations of R-peaks
        for idx, r_peak_index in enumerate(corrected_peak_inds):
            # Calculating the start and end indices for the segment
            start_index = max(r_peak_index - window_size_before, 0)
            end_index = min(r_peak_index + window_size_after + 1, len(record.p_signal))
            print(f"r_peak: {r_peak_index}, start_ind: {start_index}, end_ind: {end_index}")
            beat_annotation_index = np.where((annotation.sample >= start_index) & (annotation.sample < end_index))[
                0]
            print("beat_annotation_index:", beat_annotation_index)
            if beat_annotation_index.size > 0:
                if annotation.symbol[beat_annotation_index[0]] == 'N':
                    print(f"N index{beat_annotation_index[0]} type {annotation.symbol[beat_annotation_index[0]]}")
                    label = 0.0
                elif annotation.symbol[beat_annotation_index[0]] == 'V':
                    print(f"V index{beat_annotation_index[0]} type {annotation.symbol[beat_annotation_index[0]]}")
                    label = 1.0
                else:
                    print(f"Skipping heartbeat at index {beat_annotation_index[0]} due to unneeded label:"
                          f"{annotation.symbol[beat_annotation_index[0]]}")
                    continue
            else:
                print(f"No annotations in range {start_index} to {end_index}")
                continue

            segment = record.p_signal[start_index:end_index]

            # Inserting the label to the segment as the 0th and record # as 1st column
            segment = np.insert(segment, 0, label)
            segment = np.insert(segment, 1, i)

            with open(f"{csv_file_name_prefix}.csv", 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(segment)
    return

data_load_and_save_to_csv()

def data_load_and_save_to_csv_no_correction(total_sample_number=1, signal_start_index=124,
                              window_size_before=99, window_size_after=200,
                              save_file_path="ecg_csv_files/"):
    for i in range(signal_start_index, signal_start_index + total_sample_number):
        if i in missing_recs:
            print("record #" + str(i) + "does not exist")
            continue
        else:
            print("record #" + str(i))
            signal_loc = "../original_signals/" + str(i)
            record = wfdb.rdrecord(signal_loc, channels=[0])
            annotation = wfdb.rdann(signal_loc, 'atr')

        qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal, fs=record.fs)

        csv_file_name_prefix = save_file_path + "/" + str(i) + "_heartbeats"
        csv_file_count = 1

        for idx, r_peak_index in enumerate(qrs_inds):
            start_index = max(r_peak_index - window_size_before, 0)
            end_index = min(r_peak_index + window_size_after + 1, len(record.p_signal))
            print(f"r_peak: {r_peak_index}, start_ind: {start_index}, end_ind: {end_index}")
            beat_annotation_index = np.where((annotation.sample >= start_index) & (annotation.sample < end_index))[
                0]
            print("beat_annotation_index:", beat_annotation_index)
            if beat_annotation_index.size > 0:
                if annotation.symbol[beat_annotation_index[0]] == 'N':
                    print(f"N index {beat_annotation_index[0]} type {annotation.symbol[beat_annotation_index[0]]}")
                    label = 0.0
                elif annotation.symbol[beat_annotation_index[0]] == 'V':
                    print(f"V index {beat_annotation_index[0]} type {annotation.symbol[beat_annotation_index[0]]}")
                    label = 1.0
                else:
                    print(f"Skipping heartbeat at index {beat_annotation_index[0]} due to unneeded label:"
                          f"{annotation.symbol[beat_annotation_index[0]]}")
                    continue
            else:
                print(f"No annotations in range {start_index} to {end_index}")
                continue

            segment = record.p_signal[start_index:end_index]

            segment = np.insert(segment, 0, label)
            segment = np.insert(segment, 1, i)

            with open(f"{csv_file_name_prefix}_{csv_file_count}.csv", 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(segment)
    return

# uncomment and put correct index and number values to load the data
# data_load_and_save_to_csv_no_correction()