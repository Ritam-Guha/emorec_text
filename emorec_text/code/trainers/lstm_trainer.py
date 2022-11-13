import torch.nn

from emorec_text.code.trainers.base_trainer import Trainer
from emorec_text.code.trainers.loss import L1loss


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
            # optimizer.zero_grad()
        else:
            net = net.eval()

        i = 0
        # for param in net.parameters():
        #     print(param.grad)

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
                loss = torch.nn.MSELoss()(pred_emotion, emotion).to(self.device)
            else:
                loss += torch.nn.MSELoss()(pred_emotion, emotion).to(self.device)
            i += 1

        if type_process == "train":
            loss.backward()
            # for param in net.parameters():
            #     print(param.grad)
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            optimizer.step()

        return loss.item()

# def train_model(model):
#     train_data = get_data(type_data="train") # get the ground truth embeddings from data loader
#
#     # training
#     for i in range(50):
#         output = model(data) # output is the model emotions
#         loss = L1loss(data, output)
#         loss.backward()
#         print(loss)
#
#     test_data = get_data(type_data="test")  # get the ground truth embeddings from data loader
#     test_output = model(test_data)
#     test_loss = L1loss()test_data, test_output
#     print(test_loss)
