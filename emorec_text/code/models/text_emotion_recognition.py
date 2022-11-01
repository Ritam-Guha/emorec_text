import text2emotion
from emorec_text.code.data_utils.data_loader import DataLoader

from abc import abstractmethod
import numpy as np

import nltk
nltk.download('omw-1.4')


class Text2EmotionModel:
    def __int__(self):
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
    data_loader = DataLoader()
    data = data_loader.get_data()
    text_to_emotion = Text2EmotionModel()
    type_data = "test"
    accuracy = 0

    for text, emotions in zip(data[type_data]["text"], data[type_data]["emotion"]):
        cur_prediction = text_to_emotion.predict(text)
        correct_count = sum([i == j for i, j in zip(cur_prediction, emotions)])
        cur_accuracy = float(correct_count)/len(emotions)
        accuracy += cur_accuracy
        print(f"prediction accuracy: {np.round(cur_accuracy*100, 2)}")

    mean_accuracy = accuracy/len(data[type_data]["emotion"])
    print(f"Mean {type_data} accuracy: {np.round(mean_accuracy*100, 2)}")


if __name__ == "__main__":
    main()

