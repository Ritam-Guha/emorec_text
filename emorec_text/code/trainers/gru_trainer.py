from emorec_text.code.trainers.base_trainer import Trainer
import emorec_text.config as config
from emorec_text.code.models.loss import mse_loss_function

import torch.nn


class GRUTrainer(Trainer):
    def __init__(self,
                 type_model,
                 device):
        super().__init__(type_model,
                         device)
        self.type_model = "gru"

    def process_one_epoch(self,
                          net,
                          data_loader,
                          optimizer,
                          type_process):

        # code to process one epoch of the model
        # set the train/val/test mode
        if type_process == "train":
            net = net.train()
            optimizer.zero_grad()
        else:
            net = net.eval()

        # count the number of batches being processed
        i = 0

        # get the data from the dataloader and run the epoch
        for embedding, emotion in data_loader:
            embedding = embedding.to(self.device)
            pred_emotion = net(embedding).squeeze().to(self.device)
            emotion = emotion.squeeze()

            # mask the NA emotions
            idx_na = config.emotions.index("NA")
            mask = emotion[:, idx_na] != 1
            pred_emotion = pred_emotion[mask].to(self.device)
            emotion = emotion[mask].to(self.device)

            # if there is no emotion after masking, leave the batch
            if pred_emotion.shape[0] == 0:
                continue

            # take the loss
            if i == 0:
                loss = mse_loss_function(emotion, pred_emotion).to(self.device)
            else:
                loss += mse_loss_function(pred_emotion, emotion).to(self.device)
            i += 1

        if type_process == "train":
            loss.backward()
            optimizer.step()

        return loss.item()

