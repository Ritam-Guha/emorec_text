import os.path

import pandas as pd

import emorec_text.config as config
from emorec_text.code.data_utils.data_loader import EmotionData

import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report


class Evaluator:
    def __init__(self,
                 type_model,
                 lr,
                 n_epochs):
        self.data = EmotionData()
        self.type_model = type_model
        self.lr = lr
        self.n_epochs = n_epochs
        self.model = None

    def evaluate(self):
        acc = {}
        self.load_model()
        self.plot_training_curve()
        for type_partition in ["train", "val", "test"]:
            gt = []
            pred = []

            mean_acc = 0
            count = 0

            for embedding, emotion, video_id in zip(self.data.data[type_partition]["embedding"],
                                                    self.data.data[type_partition]["emotion"],
                                                    self.data.data[type_partition]["video_id"],):

                if emotion.shape[0] < 40:
                    pred_emotion = self.model(embedding.unsqueeze(dim=0)).squeeze()
                    mask = (emotion[:, -1] != 1)
                    emotion = emotion[mask]
                    pred_emotion = pred_emotion[mask]

                    if emotion.shape[0] != 0:
                        gt_idx = torch.argmax(emotion, dim=1)
                        pred_idx = torch.argmax(pred_emotion, dim=1)
                        cur_acc = sum(gt_idx == pred_idx).item()/len(gt_idx) * 100

                        gt += [config.emotions[idx] for idx in gt_idx]
                        pred += [config.emotions[idx] for idx in pred_idx]
                        mean_acc += cur_acc
                        count += 1

            mean_acc /= count
            print(f"mean {type_partition} acc: {mean_acc}")
            acc[type_partition] = mean_acc

            conf_mat = confusion_matrix(gt, pred, labels=config.emotions[:-1])
            sns.heatmap(conf_mat, square=True, annot=True, cmap='Blues', fmt='d', cbar=False,
                        xticklabels=config.emotions[:-1], yticklabels=config.emotions[:-1])

            emotion_wise_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1) * 100
            if type_partition == "test":
                report = classification_report(gt, pred, labels=config.emotions[:-1], target_names=config.emotions[:-1])
                with open(f"{config.BASE_PATH}/code/model_storage/{self.type_model}/lr_{self.lr}_epochs_{self.n_epochs}/"
                          f"classification_report.text", "w") as file_writer:
                    file_writer.write(report)
                    file_writer.write("\nemotion-wise classification accuracy\n")
                    for i, emotion in enumerate(config.emotions[:-1]):
                        if emotion_wise_accuracy[i] != np.float64("nan"):
                            file_writer.write(f"{emotion}: {emotion_wise_accuracy[i]}\n")

                    file_writer.write(f"\noverall accuracy: {acc['test']}")

        acc_df = pd.DataFrame()
        acc_df["train_acc"] = [acc["train"]]
        acc_df["val_acc"] = [acc["val"]]
        acc_df["test_acc"] = [acc["test"]]
        acc_df.to_csv(f"{config.BASE_PATH}/code/model_storage/{self.type_model}/lr_{self.lr}_epochs_{self.n_epochs}/"
                      f"accuracy.csv", index=False)

        return acc

    def load_model(self):
        path = f"{config.BASE_PATH}/code/model_storage/{self.type_model}/training_best.pt"
        if os.path.exists(path):
            self.model.load_weights(path)

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
            plt.savefig(f"{config.BASE_PATH}/code/model_storage/{self.type_model}/lr_{self.lr}_epochs_{self.n_epochs}/"
                        f"training_curve.jpg", dpi=400)
            # plt.show()

    def load_model(self):
        path = f"{config.BASE_PATH}/code/model_storage/{self.type_model}/lr_{self.lr}_epochs_{self.n_epochs}/" \
               f"training_best.pt"
        if os.path.exists(path):
            self.model.load_weights(path)