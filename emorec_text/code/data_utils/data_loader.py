import os
import emorec_text.config as config


class DataLoader:
    def __init__(self,
                 type_data="videos"):
        self.data_path = f"{config.BASE_PATH}/data/{type_data}"
