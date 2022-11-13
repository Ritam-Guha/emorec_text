import os.path

import emorec_text.config as config
from emorec_text.code.data_utils.data_loader import EmotionData

from torch.utils.data import DataLoader
import time


class TrainScript:
    def __init__(self,
                 type_model="lstm",
                 device="cpu"):
        self.type_model = type_model
        self.device = device
        self.data_loader = {}
        self.model = None
        self.trainer = None

    def train(self):
        self.get_model()
        self.load_model()
        self.get_trainer()
        self.get_data_loader()
        # start the training process
        start_time = time.time()
        trained_loss = self.trainer.train(model=self.model,
                                          train_loader=self.data_loader["train"],
                                          test_loader=self.data_loader["test"])
        print(f"--- {(time.time() - start_time) / 3600} hours ---")
        print(f"final loss: {trained_loss}")

    def get_data_loader(self):
        data_dict = {}
        for type_partition in ["train", "test"]:
            data_dict[type_partition] = EmotionData(type_partition=type_partition)
            print(f"{type_partition} size: {data_dict[type_partition].__len__()}")
            self.data_loader[type_partition] = DataLoader(data_dict[type_partition],
                                                          batch_size=1,
                                                          shuffle=False,
                                                          pin_memory=True,
                                                          num_workers=0)

    def get_model(self):
        pass

    def load_model(self):
        path = f"{config.BASE_PATH}/code/model_storage/{self.type_model}/training_best.pt"
        if os.path.exists(path):
            self.model.load_weights(path)

    def get_trainer(self):
        pass
