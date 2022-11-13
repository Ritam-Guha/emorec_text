import os.path

import emorec_text.config as config
from emorec_text.code.data_utils.data_loader import EmotionData

import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Evaluator:
    def __init__(self,
                 type_model):
        self.data = EmotionData()
        self.type_model = type_model

    def evaluate(self):
        acc = {}
        self.load_model()
        self.plot_training_curve()
        for type_partition in ["train", "test"]:
            mean_acc = 0
            count = 0

            for embedding, emotion in zip(self.data.data[type_partition]["embedding"],
                                          self.data.data[type_partition]["emotion"]):
                pred_emotion = self.model(embedding.unsqueeze(dim=0)).squeeze()
                mask = (emotion[:, -1] != 1)
                emotion = emotion[mask]
                pred_emotion = pred_emotion[mask]

                if emotion.shape[0] != 0:
                    gt_idx = torch.argmax(emotion, dim=1)
                    pred_idx = torch.argmax(pred_emotion, dim=1)
                    cur_acc = sum(gt_idx == pred_idx).item()/len(gt_idx) * 100
                    mean_acc += cur_acc
                    count += 1

            mean_acc /= count
            print(f"mean {type_partition} acc: {mean_acc}")
            acc[type_partition] = mean_acc

        return acc

    def plot_training_curve(self):
        path = f"{config.BASE_PATH}/code/model_storage/{self.type_model}/training_loss_curve.pickle"
        if os.path.exists(path):
            loss_curve = pickle.load(open(path, "rb"))

            for partition_type in ["train", "val"]:
                cur_loss = np.array(loss_curve[partition_type])
                cur_loss = (cur_loss - np.min(cur_loss))/(np.max(cur_loss) - np.min(cur_loss))
                plt.plot(np.arange(len(loss_curve[partition_type])), cur_loss, label=f"{partition_type}_loss")
            plt.title(f"convergence curve for {self.type_model} training")
            plt.legend(loc="upper right")
            plt.xlabel("epochs")
            plt.ylabel("normalized loss")
            plt.show()
            plt.savefig(f"{config.BASE_PATH}/code/model_storage/{self.type_model}/training_curve.jpg", dpi=400)

    def load_model(self):
        pass