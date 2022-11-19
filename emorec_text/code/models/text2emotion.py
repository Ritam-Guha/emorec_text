import torch

import emorec_text.config as config
from emorec_text.code.data_utils.data_loader import EmotionData
from emorec_text.code.utils.path_utils import create_dir

from abc import abstractmethod
import numpy as np
import pandas as pd
import text2emotion

import nltk
nltk.download('omw-1.4')


def evaluate(data,
             model,
             type_partition="test"):

    # initialize the necessary variables
    create_dir(f"results/{model.name}")
    accuracy = 0
    accuracy_list = []
    video_id_list = data[type_partition]["video_id"]
    count = 0

    for text, emotions in zip(data[type_partition]["text"], data[type_partition]["emotion"]):
        cur_prediction = model.predict(text)

        # we are not counting NA emotions for training or testing, so mask them
        mask_emotions = [emotion == "NA" for emotion in emotions]
        mask_index = list(np.nonzero(mask_emotions)[0])
        all_index = list(np.arange(len(emotions)))
        selected_index = list(set(all_index) - set(mask_index))

        if len(selected_index) == 0:
            video_id_list.remove(video_id_list[count])
            continue

        # take the selected emotions
        emotions = [emotions[i] for i in selected_index]
        cur_prediction = [cur_prediction[i] for i in selected_index]

        # check the correct predictions
        correct_count = sum([i == j for i, j in zip(cur_prediction, emotions) if j != np.nan])
        cur_accuracy = (float(correct_count) / len(emotions)) * 100

        accuracy_list.append(cur_accuracy)
        accuracy += cur_accuracy
        print(f"prediction accuracy for video {count}: {np.round(cur_accuracy, 2)}")
        count += 1

    mean_accuracy = accuracy/count
    print(f"Mean {type_partition} accuracy: {np.round(mean_accuracy, 2)}")
    video_id_list.append("Mean")
    accuracy_list.append(mean_accuracy)

    df = pd.DataFrame()
    df["video_id"] = video_id_list
    df["accuracy"] = accuracy_list
    df.to_csv(f"{config.BASE_PATH}/results/{model.name}/summary.csv", index=False)


class Text2EmotionModel:
    def __init__(self):
        self.name = "text2emotion"

    @abstractmethod
    def predict(self,
                text_list):

        list_emotion_predictions = []

        for text in text_list:
            prediction_prob = text2emotion.get_emotion(text)
            max_key = max(prediction_prob, key=prediction_prob.get)
            list_emotion_predictions.append(max_key.lower())

        return list_emotion_predictions


def main():
    data_loader = EmotionData(type_data="text")
    data = data_loader.get_data()
    model = Text2EmotionModel()
    type_partition = "test"
    evaluate(data,
             model,
             type_partition=type_partition)


if __name__ == "__main__":
    main()

