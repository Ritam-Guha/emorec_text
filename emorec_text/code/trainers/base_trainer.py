import emorec_text.config as config
from emorec_text.code.utils.path_utils import create_dir
from emorec_text.code.losses.get_loss import get_loss
from emorec_text.code.evaluation.get_evaluator import get_evaluator

import copy
import torch
import pickle
import warnings

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self,
                 type_model,
                 device="cpu"):
        # initialize the variables
        self.type_model = type_model
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.lr = None
        self.n_epochs = None
        self.loss = None
        self.evaluator = None
        create_dir(f"code/model_storage/{self.type_model}")
        self.save = f"{config.BASE_PATH}/code/model_storage/{self.type_model}"

    def train(self,
              model,
              train_loader,
              val_loader,
              test_loader,
              n_epochs=1000,
              lr=1e-5,
              type_loss="mse"):

        self.lr = lr
        self.n_epochs = n_epochs
        self.save += f"/lr_{self.lr}_epochs_{self.n_epochs}"
        create_dir(self.save.replace(config.BASE_PATH, ""))
        self.evaluator = get_evaluator(type_model=self.type_model)(lr=lr,
                                                                   n_epochs=n_epochs)

        print(f"type loss: {type_loss}")

        data_loader = {"train": train_loader,
                       "val": val_loader,
                       "test": test_loader}

        # initialize the loss function
        self.loss = get_loss(type_loss=type_loss)

        # initialize the optimizer
        self.optimizer = torch.optim.SGD(model.parameters(),
                                         lr=self.lr)

        # set the scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=50,
                                                         gamma=0.5,
                                                         last_epoch=-1)

        prev_loss = 1e33
        loss_curve = {"train": [], "val": []}

        for epoch in range(n_epochs):
            # compute the train and validation loss after processing an epoch
            train_loss = self.process_one_epoch(net=model,
                                                data_loader=data_loader["train"],
                                                optimizer=self.optimizer,
                                                type_process="train")

            val_loss = self.process_one_epoch(net=model,
                                              data_loader=data_loader["val"],
                                              optimizer=self.optimizer,
                                              type_process="val")
            print(f"epoch: {epoch}, train_loss: {train_loss}, val loss: {val_loss}")

            # add the losses to the curve for plotting
            loss_curve["train"].append(train_loss)
            loss_curve["val"].append(val_loss)

            # if validation loss is better, checkpoint the current model
            if val_loss < prev_loss:
                prev_loss = val_loss
                self.checkpoint(epoch=epoch,
                                model=copy.deepcopy(model),
                                optimizer=copy.deepcopy(self.optimizer),
                                lr_sched=copy.deepcopy(self.scheduler))

            self.scheduler.step(val_loss)

        # finally check the loss on the test data
        test_loss = self.process_one_epoch(net=model,
                                           data_loader=data_loader["test"],
                                           optimizer=self.optimizer,
                                           type_process="test")
        print(test_loss)
        pickle.dump(loss_curve,
                    open(f"{config.BASE_PATH}/code/model_storage/{self.type_model}/training_loss_curve.pickle", "wb"))

        # get the final evaluation
        acc = self.evaluator.evaluate()

        return test_loss

    def checkpoint(self,
                   epoch,
                   model,
                   optimizer,
                   lr_sched):
        # checkpoint storage for the model
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_sched}

        torch.save(checkpoint, f"{self.save}/training_best.pt")

    def process_one_epoch(self,
                          net,
                          data_loader,
                          optimizer,
                          type_process):
        pass
