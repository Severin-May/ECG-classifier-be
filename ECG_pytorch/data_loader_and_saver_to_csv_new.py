import wfdb
from wfdb import processing
import numpy as np
import csv
import matplotlib.pyplot as plt

""" some record numbers are not present in the db, hence skipping those"""
MISSING_RECS = [110, 120, 204, 206, 211, 216, 218, 224, 225, 226, 227, 229]
RECS_FOLDER_PATH = "../original_signals/"
SAVE_FILE_PATH = "ecg_csv_files/"
""" window size can be adjusted """
WINDOW_SIZE_BEFORE = 99
WINDOW_SIZE_AFTER = 200

def data_load_and_save_to_csv(total_sample_number=25, signal_start_index=100):
    cnt = 0
    for i in range(signal_start_index, signal_start_index + total_sample_number):
        with open("data_load_and_save_to_csv.log", 'w') as file:

            if i in MISSING_RECS:
                file.write(f"RECORD #{i} does not exist\n\n")
                continue
            else:
                signal_loc = RECS_FOLDER_PATH + str(i)
                record = wfdb.rdrecord(signal_loc, channels=[0])
                file.write(f"writing RECORD #{i}\n\n")
                annotation = wfdb.rdann(signal_loc, 'atr')

            qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal, fs=record.fs)
            csv_file_name_prefix = SAVE_FILE_PATH + "/" + str(i) + "_heartbeats"

            for idx, r_peak in enumerate(qrs_inds):
                start_index = max(0, r_peak - WINDOW_SIZE_BEFORE)
                end_index = min(len(record.p_signal), r_peak + WINDOW_SIZE_AFTER)
                file.write(f"r_peak: {r_peak}, start_ind: {start_index}, end_ind: {end_index}\n")
                if end_index - start_index == WINDOW_SIZE_BEFORE + WINDOW_SIZE_AFTER:
                    beat_annotation_index = np.where((annotation.sample >= start_index) &
                                                     (annotation.sample < end_index))[0]
                    file.write(f"beat_annotation_index: {beat_annotation_index}\n")

                    if beat_annotation_index.size > 0:
                        if annotation.symbol[beat_annotation_index[0]] == 'N':
                            file.write(
                                f"N index {annotation.sample[beat_annotation_index[0]]} type {annotation.symbol[beat_annotation_index[0]]}\n")
                            heartbeat_segment = record.p_signal[start_index-1:end_index]
                            heartbeat_segment = np.insert(heartbeat_segment, 0, 0.0)
                        elif annotation.symbol[beat_annotation_index[0]] == 'V':
                            file.write(
                                f"V index {annotation.sample[beat_annotation_index[0]]} type {annotation.symbol[beat_annotation_index[0]]}\n")
                            heartbeat_segment = record.p_signal[start_index-1:end_index]
                            heartbeat_segment = np.insert(heartbeat_segment, 0, 1.0)
                        else:
                            file.write(f"Skipping heartbeat at index {annotation.sample[beat_annotation_index[0]]} due to unneeded label:"
                                  f"{annotation.symbol[beat_annotation_index[0]]}\n")
                            continue
                        """ uncomment to plot the first 10 beats """
                        # if cnt < 10:
                        #     plt.plot(heartbeat_segment)
                        #     plt.show()
                        #     cnt = cnt + 1
                        if np.any(np.isnan(heartbeat_segment)):
                            print("There is Nan")
                    else:
                        file.write(f"No annotations in range {start_index} to {end_index}\n")
                        continue
                else:
                    file.write(f"Error: Incomplete segmentation from {start_index} to {end_index}\n")
                    continue
                heartbeat_segment = np.insert(heartbeat_segment, 1, i)
                with open(f"{csv_file_name_prefix}.csv", 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(heartbeat_segment)
    print("Done with loading!\n")

""" uncomment and put the correct #of records and start index above to load the data from wfdb """
# data_load_and_save_to_csv()
