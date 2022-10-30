import emorec_text.config as config

import os
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self,
                 seed=0,
                 train_percent=0.7):

        self.data_path = f"{config.BASE_PATH}/data/text_emotion"
        self.train_percent = train_percent
        self.seed = seed
        self.data = self.read_data()

    def read_data(self):
        list_files = os.listdir(self.data_path)
        idx_list = np.arange(len(list_files))
        train_size = int(self.train_percent * len(list_files))

        np.random.seed(self.seed)
        np.random.shuffle(idx_list)

        train_idx = idx_list[:train_size]
        test_idx = idx_list[train_size:]

        train_list = [list_files[i] for i in train_idx]
        test_list = [list_files[i] for i in test_idx]

        data = {"train": {"text": [], "emotion": []},
                "test": {"text": [], "emotion": []}}

        # train
        for video_id in train_list:
            df_emotion = pd.read_csv(f"{config.BASE_PATH}/data/text_emotion/{video_id}")
            data["train"]["text"].append(list(df_emotion["text"]))
            data["train"]["emotion"].append(list(df_emotion["dominant_emotion"]))

        # test
        for video_id in test_list:
            df_emotion = pd.read_csv(f"{config.BASE_PATH}/data/text_emotion/{video_id}")
            data["test"]["text"].append(list(df_emotion["text"]))
            data["test"]["emotion"].append(list(df_emotion["dominant_emotion"]))

        print(f"train size: {len(train_list)}")
        print(f"test size: {len(test_list)}")
        return data

    def get_data(self):
        return self.data


def main():
    data_loader = DataLoader()


if __name__ == "__main__":
    main()
