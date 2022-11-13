import emorec_text.config as config
from emorec_text.code.utils.path_utils import create_dir

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import string
import re
import torch
import nltk
import pickle
from torch.utils.data import Dataset
import time

# download nltk data elements
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


class EmotionData(Dataset):
    def __init__(self,
                 device="cpu",
                 seed=0,
                 train_percent=0.7,
                 type_data="text",
                 type_partition="test",
                 save_data=False):

        assert(type_data in ["text", "embedding"])
        self.data_path = f"{config.BASE_PATH}/data/text_emotion"
        self.train_percent = train_percent
        self.seed = seed
        self.device = device
        self.type_data = type_data
        self.save_data = save_data
        self.type_partition = type_partition

        if os.path.exists(f"{config.BASE_PATH}/data/object_storage/data.pickle"):
            self.data = pickle.load(open(f"{config.BASE_PATH}/data/object_storage/data.pickle", "rb"))
        else:
            self.data = self.read_data()

        if self.save_data:
            create_dir("data/object_storage")
            pickle.dump(self.data, open(f"{config.BASE_PATH}/data/object_storage/data.pickle", "wb"))

    def read_data(self):
        list_files = os.listdir(self.data_path)
        idx_list = np.arange(len(list_files))
        train_size = int(self.train_percent * len(list_files))
        val_size = int(len(list_files) - train_size)//2

        np.random.seed(self.seed)
        np.random.shuffle(idx_list)

        train_idx = idx_list[:train_size]
        val_idx = idx_list[train_size:train_size+val_size]
        test_idx = idx_list[train_size+val_size:]

        train_list = [list_files[i] for i in train_idx]
        val_list = [list_files[i] for i in val_idx]
        test_list = [list_files[i] for i in test_idx]

        list_data_files = {"train": train_list, "val": val_list, "test": test_list}

        data = {"train": {"text": [], "emotion": [], "embedding": [], "video_id": []},
                "val": {"text": [], "emotion": [], "embedding": [], "video_id": []},
                "test": {"text": [], "emotion": [], "embedding": [], "video_id": []}}

        print(f"train size: {len(train_list)}")
        print(f"val size: {len(val_list)}")
        print(f"test size: {len(test_list)}")

        for partition_type in ["train", "val", "test"]:
            for i, video_id in enumerate(list_data_files[partition_type]):
                df_emotion = pd.read_csv(f"{config.BASE_PATH}/data/text_emotion/{video_id}")
                if df_emotion.isnull().any().any():
                    continue

                data[partition_type]["video_id"].append(video_id)
                data[partition_type]["text"].append(list(df_emotion["text"]))
                list_emotions = []
                for emotion in df_emotion["dominant_emotion"]:
                    list_emotions.append([(i == emotion) * 1 for i in config.emotions])
                data[partition_type]["emotion"].append(torch.DoubleTensor(list_emotions))
                if self.type_data == "embedding":
                    cur_text_list = []
                    for text in list(df_emotion["text"]):
                        cur_text_list.append(self.preprocess_string(text))
                    cur_embedding_list = self.string_encoding(cur_text_list)

                    cur_embedding_tensor = torch.DoubleTensor(cur_embedding_list)
                    data[partition_type]["embedding"].append(cur_embedding_tensor)

                print(f"{partition_type} video file {i} loading done...")
            # data[partition_type]["emotion"] = torch.DoubleTensor(data[partition_type]["emotion"])

        return data

    def get_data(self):
        return self.data

    @staticmethod
    def preprocess_string(text):
        # remove punctuations
        text = "".join([i for i in text if i not in string.punctuation])

        # lower case conversion
        text = text.lower()

        # tokenization
        tokens = re.split('\W+', text)

        # stop word removal
        stopwords = nltk.corpus.stopwords.words('english')
        tokens = [token for token in tokens if token not in stopwords]

        # lemmatization
        wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
        final_tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]

        # combined the tokens into a string
        mod_text = " ".join([i for i in final_tokens])

        return mod_text

    def string_encoding(self,
                        text):

        # use the bert encoding
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens').to(self.device)
        embedded_data = sbert_model.encode(text)
        return embedded_data

    def __len__(self):
        return len(self.data[self.type_partition]["embedding"])

    def __getitem__(self, idx):
        return self.data[self.type_partition]["embedding"][idx], self.data[self.type_partition]["emotion"][idx]


def main():
    # obj = pickle.load(open(f"{config.BASE_PATH}/data/object_storage/data_loader.pickle", "rb"))
    start_time = time.time()
    data_loader = EmotionData(type_data="embedding",
                              save_data=True)
    print(f"--- {(time.time() - start_time) / 3600} hours ---")


if __name__ == "__main__":
    main()
