"""
    Builder class for creating configuration dictionaries for the data loader.

    Usage:
    config_builder = DataLoaderConfigBuilder()
    config = config_builder.set_recs_folder_path("../original_signals/") \
                            .set_save_file_path("ecg_csv_files/") \
                            .set_window_size_before(99) \
                            .set_window_size_after(200) \
                            .set_missing_recs([110, 120, 204, 206, 211, 216, 218, 224, 225, 226, 227, 229]) \
                            .build()
"""


class DataLoaderConfigBuilder:
    """
        Initialize the DataLoaderConfigBuilder with default configuration values.
    """

    def __init__(self):
        self.config = {
            'missing_recs': [110, 120, 204, 206, 211, 216, 218, 224, 225, 226, 227, 229],
            'recs_folder_path': "../original_signals/",
            'save_file_path': "ecg_csv_files/",
            'window_size_before': 99,
            'window_size_after': 200
        }

    def set_missing_recs(self, missing_recs):
        self.config['missing_recs'] = missing_recs
        return self

    def set_recs_folder_path(self, recs_folder_path):
        self.config['recs_folder_path'] = recs_folder_path
        return self

    def set_save_file_path(self, save_file_path):
        self.config['save_file_path'] = save_file_path
        return self

    def set_window_size_before(self, window_size_before):
        self.config['window_size_before'] = window_size_before
        return self

    def set_window_size_after(self, window_size_after):
        self.config['window_size_after'] = window_size_after
        return self

    def build(self):
        return self.config
