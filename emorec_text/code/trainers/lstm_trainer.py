import torch.nn

from emorec_text.code.trainers.base_trainer import Trainer
from emorec_text.code.models.loss import cross_entropy_loss_function


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

        if type_process == "train":
            net = net.train()
            optimizer.zero_grad()
        else:
            net = net.eval()

        i = 0
        for embedding, emotion in data_loader:
            embedding = embedding.to(self.device)
            pred_emotion = net(embedding).squeeze().to(self.device)
            emotion = emotion.squeeze()
            mask = emotion[:, -1] != 1
            pred_emotion = pred_emotion[mask].to(self.device)
            emotion = emotion[mask].to(self.device)

            if pred_emotion.shape[0] == 0:
                continue

            if i == 0:
                loss = torch.nn.MSELoss()(emotion, pred_emotion).to(self.device)
            else:
                loss += torch.nn.MSELoss()(emotion, pred_emotion).to(self.device)

            # if i == 0:
            #     loss = cross_entropy_loss_function(emotion, pred_emotion).to(self.device)
            # else:
            #     loss += cross_entropy_loss_function(emotion, pred_emotion).to(self.device)
            i += 1

        if type_process == "train":
            loss.backward()
            optimizer.step()

        return loss.item()
