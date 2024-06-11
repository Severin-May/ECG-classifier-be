import os
import pandas as pd
import numpy as np

TEST_FILE_RECORDS = [106, 116, 215, 228, 231, 212, 112, 107, 201]
REC_STATISTICS_FILE = "record_statistics.txt"


class FileProcessor:
    def process(self, file_path):
        raise NotImplementedError


class CSVProcessor(FileProcessor):
    def process(self, file_path):
        df = pd.read_csv(file_path, header=None)
        healthy_beats = df.loc[df.iloc[:, 0] == 0.0, df.columns[0:]].to_numpy()
        unhealthy_beats = df.loc[df.iloc[:, 0] == 1.0, df.columns[0:]].to_numpy()

        with open(REC_STATISTICS_FILE, 'w') as file:
            file.write(f"File:{file_path}, #healthy: {len(healthy_beats)},     #unhealthy: {len(unhealthy_beats)}\n")

        return healthy_beats, unhealthy_beats


class FileProcessorContext:
    def __init__(self, file_processor):
        self.file_processor = file_processor

    def process_file(self, file_path):
        return self.file_processor.process(file_path)


""" checks if record file is among test record files """


def is_test_file(filename):
    return any(str(rec) in filename for rec in TEST_FILE_RECORDS)


""" iterates through .csv files and saves the content as Tensors """


def process_multiple_files(folder_path, file_processor_context):
    all_healthy_beats = np.empty((0, 302))
    all_unhealthy_beats = np.empty((0, 302))
    all_test_beats = np.empty((0, 302))
    total_number_healthy_beats = 0
    total_number_unhealthy_beats = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)

            healthy_beats, unhealthy_beats = file_processor_context.process_file(file_path)
            total_number_healthy_beats += len(healthy_beats)
            total_number_unhealthy_beats += len(unhealthy_beats)

            if is_test_file(file_name):
                all_test_beats = np.concatenate((all_test_beats, healthy_beats), axis=0)
                all_test_beats = np.concatenate((all_test_beats, unhealthy_beats), axis=0)
            else:
                all_healthy_beats = np.concatenate((all_healthy_beats, healthy_beats), axis=0)
                all_unhealthy_beats = np.concatenate((all_unhealthy_beats, unhealthy_beats), axis=0)

    print(f"#total_healthy_beats: {total_number_healthy_beats}, #total_unhealthy_beats: {total_number_unhealthy_beats}")
    return all_healthy_beats, all_unhealthy_beats, all_test_beats

# csv_processor = CSVProcessor()
# csv_processor_context = FileProcessorContext(csv_processor)
# all_healthy_beats, all_unhealthy_beats, all_test_beats = process_multiple_files("../ecg_csv_files",
#                                                                                 csv_processor_context)
