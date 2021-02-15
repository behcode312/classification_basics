import collections
import config
import copy
import numpy as np
import io
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field
from pathlib import Path
from torchvision import datasets, models, transforms
from typing import (List, Dict)
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class DataModelParams():
    """
    This gets and sets dataset parameters.
    
    Attributes
    mean_dataset: list
        RGB mean of taining set
    std_dataset: list
        RGB std of training set

    Methods
        data_param()
        load dataloader
    """
    mean_dataset: List[float] = field(default_factory=list)
    std_dataset: List[float] = field(default_factory=list)
 
    def  __post_init__(self):
        self.mean_dataset = [0.6845, 0.5535, 0.6924]
        self.std_dataset = [0.1848, 0.2138, 0.1596]

    def data_param(self):
        """
        set pytorch dataloader parameters
        Args:
            data_dir: I have to get this argument from config

        Returns:
            A dictionary of parameter, which the most important one is dataset_dataloader.
            
        """
        data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(hue=0.09, saturation=0.15),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean_dataset, self.std_dataset),
                ]
            ),
            "val": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(self.mean_dataset, self.std_dataset)]
            ),
        }

        dataset_folder = {
            x: datasets.ImageFolder(Path(config.args.data_dir, x), data_transform[x])
            for x in ["train", "val"]
        }
        print(f"{dataset_folder['train'].class_to_idx}")

        dataset_dataloader = {
            x: torch.utils.data.DataLoader(
                dataset_folder[x], batch_size=4, shuffle=True, num_workers=8
            )
            for x in ["train", "val"]
        }

        dataset_size = {x: len(dataset_folder[x]) for x in ["train", "val"]}
        print(f"{dataset_size}")

        class_names = dataset_folder["train"].classes

        dataparams_dict = {
            "data_transform": data_transform,
            "data_dir": config.args.data_dir,
            "dataset_folder": dataset_folder,
            "dataset_dataloader": dataset_dataloader,
            "dataset_size": dataset_size,
            "class_names": class_names,
            "device": config.args.device,
        }
        return dataparams_dict

    def model_param(self):
        """
        Everything model parameters are needed should be set in this method.

        Returns
            model: the model that we want to train
            criterion: the weighted criterion that we are using for our imbalanced dataset
            optimizer: optimizer that is needed during training
            scheudler: the schdudler that would update learning rate during training1
        """
        resnet_model = models.resnet18(pretrained=True)

        # parameters of the pretrained model is set to requires_grad=True
        # we cahng change it if any layer needs to be freezed
        for param in resnet_model.parameters():
            param.requires_grad = True

        # changing fc layer to the number of our dataset classes
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_ftrs, 5)

        weight = torch.tensor(config.args.weight_loss)
        weight = weight.cuda(non_blocking=True)
        criterion = nn.CrossEntropyLoss(weight=weight, reduction="mean")
        optimizer = optim.Adam(resnet_model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

        return resnet_model, criterion, optimizer, scheduler

@dataclass
class TrainVal():
    """
    Train validate and test model.

    Attributes
        model: the classifier model for training
        criterion: criterion of the model
        oprimizer: optimizer 
        scheulder: scheduler
        num_epochs: number of training epochs
        device: training device_
        dataset_size: size of the training and validation data
        dataloader: dataloader that is needed for classification
        save_path: path that model checkpoints and results are saved.
    
    Methods
        train_model()
            train the model with given parameters
        test_model()
            test the model on the given checkpoints
        resume_model()
            resume the model from the given checkpoints and resume training.
    """
    model: torchvision.models.resnet.ResNet
    criterion: torch.nn.modules.loss.CrossEntropyLoss
    optimizer: torch.optim.Adam
    scheduler: torch.optim.lr_scheduler.ExponentialLR
    num_epochs: int
    device: int 
    dataset_size: Dict[str, int]
    dataloader: Dict[str, torch.utils.data.DataLoader]
    save_path: Path
    
    def train_model(self):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        loss_history = collections.defaultdict(int)
        acc_history = collections.defaultdict(int)
        lr = collections.defaultdict(int)

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}/{self.num_epochs - 1}")
            # each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                current_loss = 0.0
                current_corrects = 0

                for inputs, labels in self.dataloader[phase]:
                    inputs = inputs.cuda(non_blocking=True)
                    labels = labels.to(non_blocking=True)

                    # set gradient to zero
                    self.optimizer.zero_grad()

                    # Forwarding pass
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # Backward and optimize
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()
                            lr[epoch] = self.optimizer.param_groups[0]["lr"]

                    # keep the loss statistical information
                    current_loss += loss.item() * inputs.size(0)
                    current_corrects += torch.sum(preds == labels.data)

                epoch_loss = current_loss / self.dataset_size[phase]
                epoch_acc = current_corrects.double() / self.dataset_size[phase]
                
                # we can use these two dictionaries instead of tensorboard
                # loss_history[(epoch, phase)] = epoch_loss
                # acc_history[(epoch, phase)] = epoch_acc
                if phase == "train":
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('Acc/train', epoch_acc, epoch)
                    writer.add_scalar('lr/train', self.optimizer.param_groups[0]["lr"])
                if phase == "val":
                    writer.add_scalar('Loss/val', epoch_loss, epoch)
                    writer.add_scalar('Acc/val', epoch_acc, epoch)

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                # keep a copy of the model only if the accuracy on the validation set has improved
                if phase == "val" and epoch_acc > best_acc:
                    print("saving checkpoints")
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": best_model_wts,
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "loss": loss,
                            "best_acc": best_acc,
                            "loss_history": loss_history,
                            "acc_history": acc_history,
                        },
                        str(self.save_path) + "/epoch" + str(epoch) + ".pth",
                    )
                # updating scheduler 
                self.scheduler.step()
        time_since = time.time() - since
        print(f"Training complete in {time_since//60}m {time_since % 60}s")
        print(f"Best val Acc: {best_acc:.4f}")

        # loading best model weights and return it
        self.model.load_state_dict(best_model_wts)
        last_model_wts = copy.deepcopy(self.model.state_dict())
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": last_model_wts,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss": loss,
                "best_acc": best_acc,
                "loss_history": loss_history,
                "acc_history": acc_history,
            },
            str(self.save_path) + "/epoch" + str(epoch) + ".pth",
        )
        return self.model

if __name__ == "__main__":
    training_phase = True

    if training_phase:
        params = DataModelParams()
        dataparams = params.data_param()
        
        resnet_model, criterion, optimizer, scheduler = params.model_param()
        resnet_model = resnet_model.cuda()
        
        train_model = TrainVal(
            model = resnet_model,
            criterion = criterion,
            optimizer = optimizer,
            scheduler = scheduler,
            num_epochs = 100,
            device = dataparams["device"],
            dataset_size = dataparams["dataset_size"],
            dataloader = dataparams["dataset_dataloader"],
            save_path = config.args.checkpoints,
        )

        model = train_model.train_model()
