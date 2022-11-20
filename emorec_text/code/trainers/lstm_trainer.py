import torch.nn

from emorec_text.code.trainers.base_trainer import Trainer


class LSTMTrainer(Trainer):
    def __init__(self,
                 type_model,
                 device):
        super().__init__(type_model,
                         device)
        self.type_model = "lstm"

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
            mask = emotion[:, -1] != 1
            pred_emotion = pred_emotion[mask].to(self.device)
            emotion = emotion[mask].to(self.device)

            # if there is no emotion after masking, leave the batch
            if pred_emotion.shape[0] == 0:
                continue

            # calculate the loss
            if i == 0:
                loss = self.loss(emotion, pred_emotion).to(self.device)
            else:
                loss += self.loss(emotion, pred_emotion).to(self.device)
            i += 1

        # perform the backpropagation
        if type_process == "train":
            loss.backward()
            optimizer.step()

        return loss.item()
