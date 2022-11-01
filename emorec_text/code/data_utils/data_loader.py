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

# download nltk data elements
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


class DataLoader:
    def __init__(self,
                 seed=0,
                 train_percent=0.7,
                 type_data="text",
                 save_object=False):

        assert(type_data in ["text", "embedding"])
        self.data_path = f"{config.BASE_PATH}/data/text_emotion"
        self.train_percent = train_percent
        self.seed = seed
        self.type_data = type_data
        self.save_object = save_object
        self.data = self.read_data()

        if self.save_object:
            create_dir("data/object_storage")
            pickle.dump(self, open(f"{config.BASE_PATH}/data/object_storage/data_loader.pickle", "wb"))

    def read_data(self):
        list_files = os.listdir(self.data_path)
        idx_list = np.arange(len(list_files))
        train_size = int(self.train_percent * len(list_files))

        np.random.seed(self.seed)
        np.random.shuffle(idx_list)

        train_idx = idx_list[:train_size]
        test_idx = idx_list[train_size:]

        train_list = [list_files[i] for i in train_idx][:2]
        test_list = [list_files[i] for i in test_idx][:2]

        list_data_files = {"train": train_list, "test": test_list}

        data = {"train": {"text": [], "emotion": [], "embedding": []},
                "test": {"text": [], "emotion": [], "embedding": []}}

        print(f"train size: {len(train_list)}")
        print(f"test size: {len(test_list)}")

        for partition_type in ["train", "test"]:
            for i, video_id in enumerate(list_data_files[partition_type]):
                df_emotion = pd.read_csv(f"{config.BASE_PATH}/data/text_emotion/{video_id}")
                data[partition_type]["text"].append(list(df_emotion["text"]))
                data[partition_type]["emotion"].append(list(df_emotion["dominant_emotion"]))
                if self.type_data == "embedding":
                    cur_embedding_list = []
                    for text in list(df_emotion["text"]):
                        cur_embedding_list.append(self.string_encoding(self.preprocess_string(text)))

                    cur_embedding_tensor = torch.DoubleTensor(cur_embedding_list)

                data[partition_type]["embedding"].append(cur_embedding_tensor)
                print(f"{partition_type} video file {i} done...")

        return data

    def get_data(self):
        return self.data

    def preprocess_string(self,
                          text):
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
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        embedded_data = sbert_model.encode(text)
        return embedded_data


def main():
    data_loader = DataLoader(type_data="embedding",
                             save_object=True)


if __name__ == "__main__":
    main()
