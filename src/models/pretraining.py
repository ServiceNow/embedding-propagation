import numpy as np
import os
import torch
import torch.nn.functional as F
from src.tools.meters import BasicMeter
from embedding_propagation import EmbeddingPropagation, LabelPropagation
from .base_wrapper import BaseWrapper
from src.modules.distances import prototype_distance

class PretrainWrapper(BaseWrapper):
    """Trains a model using an episodic scheme on multiple GPUs"""

    def __init__(self, model, nclasses, exp_dict):
        """ Constructor
        Args:
            model: architecture to train
            nclasses: number of output classes
            exp_dict: reference to dictionary with the hyperparameters
        """
        super().__init__()
        self.model = model
        self.exp_dict = exp_dict 
        self.ngpu = self.exp_dict["ngpu"]
        self.embedding_propagation = EmbeddingPropagation()
        self.label_propagation = LabelPropagation()
        self.model.add_classifier(nclasses, modalities=0)
        self.nclasses = nclasses
        

        if self.exp_dict["rotation_weight"] > 0:
            self.model.add_classifier(4, "classifier_rot")

        # Add optimizers here
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                            lr=self.exp_dict["lr"],
                                            momentum=0.9,
                                            weight_decay=self.exp_dict["weight_decay"], 
                                            nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode="min" if "loss" in self.exp_dict["target_loss"] else "max",
                                                                    patience=self.exp_dict["patience"])
        self.model.cuda()
        if self.ngpu > 1:
            self.parallel_model = torch.nn.DataParallel(self.model, device_ids=list(range(self.ngpu)))

    def get_logits(self, embeddings, support_size, query_size, nclasses):
        """Computes the logits from the queries of an episode
        
        Args:
            embeddings (torch.Tensor): episode embeddings
            support_size (int): size of the support set
            query_size (int): size of the query set
            nclasses (int): number of classes
        
        Returns:
            torch.Tensor: logits
        """
        b, c = embeddings.size()

        if self.exp_dict["embedding_prop"] == True:
            embeddings = self.embedding_propagation(embeddings)
        if self.exp_dict["distance_type"] == "labelprop":
            support_labels = torch.arange(nclasses, device=embeddings.device).view(1, nclasses).repeat(support_size, 1).view(support_size, nclasses)
            unlabeled_labels = nclasses * torch.ones(query_size * nclasses, dtype=support_labels.dtype, device=support_labels.device).view(query_size, nclasses)
            labels = torch.cat([support_labels, unlabeled_labels], 0).view(-1)
            logits = self.label_propagation(embeddings, labels, nclasses)
            logits = logits.view(-1, nclasses, nclasses)[support_size:(support_size + query_size), ...].view(-1, nclasses)

        elif self.exp_dict["distance_type"] == "prototypical":
            embeddings = embeddings.view(-1, nclasses, c)
            support_embeddings = embeddings[:support_size]
            query_embeddings = embeddings[support_size:]
            logits = prototype_distance(support_embeddings.view(-1, c), 
                                        query_embeddings.view(-1, c),
                                        support_labels.view(-1))
        return logits
    
    def train_on_batch(self, batch):
        """Computes the loss of a batch
        
        Args:
            batch (tuple): Inputs and labels
        
        Returns:
            loss: Loss on the batch
        """
        x, y, r = batch
        y = y.cuda(non_blocking=True).view(-1)
        r = r.cuda(non_blocking=True).view(-1)
        k, n, c, h, w = x.size()
        x = x.view(n*k, c, h, w).cuda(non_blocking=True)
        if self.ngpu > 1:
            embeddings = self.parallel_model(x, is_support=True)
        else:
            embeddings = self.model(x, is_support=True)
        b, c = embeddings.size()
                    
        loss = 0
        if self.exp_dict["rotation_weight"] > 0:
            rot = self.model.classifier_rot(embeddings) 
            loss += F.cross_entropy(rot, r) * self.exp_dict["rotation_weight"]

        if self.exp_dict["embedding_prop"] == True:
            embeddings = self.embedding_propagation(embeddings)
        logits = self.model.classifier(embeddings)
        loss += F.cross_entropy(logits, y) * self.exp_dict["cross_entropy_weight"]
        return loss

    def val_on_batch(self, batch):
        """Computes the loss and accuracy on a validation batch
        
        Args:
            batch (dict): Episode dict
        
        Returns:
            tuple: loss and accuracy of the episode 
        """
        nclasses = batch["nclasses"]
        query_size = batch["query_size"]

        logits = self.predict_on_batch(batch)

        query_labels = torch.arange(nclasses, device=logits.device).view(1, nclasses).repeat(query_size, 1).view(-1)
        loss = F.cross_entropy(logits, query_labels)
        accuracy = float(logits.max(-1)[1].eq(query_labels).float().mean())
        
        return loss, accuracy

    def predict_on_batch(self, batch):
        """Computes the logits of an episode
        
        Args:
            batch (dict): episode dict
        
        Returns:
            tensor: logits for the queries of the current episode
        """
        nclasses = batch["nclasses"]
        support_size = batch["support_size"]
        query_size = batch["query_size"]
        k = (support_size + query_size)
        c = batch["channels"]
        h = batch["height"]
        w = batch["width"]

        tx = batch["support_set"].view(support_size, nclasses, c, h, w).cuda(non_blocking=True)
        vx = batch["query_set"].view(query_size, nclasses, c, h, w).cuda(non_blocking=True)
        x = torch.cat([tx, vx], 0)
        x = x.view(-1, c, h, w).cuda(non_blocking=True)

        if self.ngpu > 1:
            embeddings = self.parallel_model(x, is_support=True)
        else:
            embeddings = self.model(x, is_support=True)
        b, c = embeddings.size()
        
        return self.get_logits(embeddings, support_size, query_size, nclasses)

    def train_on_loader(self, data_loader, max_iter=None, debug_plot_path=None):
        """Iterate over the training set
        
        Args:
            data_loader (torch.utils.data.DataLoader): a pytorch dataloader
            max_iter (int, optional): Max number of iterations if the end of the dataset is not reached. Defaults to None.
        
        Returns:
            metrics: dictionary with metrics of the training set
        """
        self.model.train()
        train_loss_meter = BasicMeter.get("train_loss").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(data_loader):
            self.optimizer.zero_grad()
            loss = self.train_on_batch(batch)
            train_loss_meter.update(float(loss), 1)
            loss.backward()
            self.optimizer.step()
            if batch_idx + 1 == max_iter:
                break
        return {"train_loss": train_loss_meter.mean()}
        

    @torch.no_grad()
    def val_on_loader(self, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        val_loss_meter = BasicMeter.get("val_loss").reset()
        val_accuracy_meter = BasicMeter.get("val_accuracy").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, _data in enumerate(data_loader):
            batch = _data[0]
            loss, accuracy = self.val_on_batch(batch)
            val_loss_meter.update(float(loss), 1)
            val_accuracy_meter.update(float(accuracy), 1)
        loss = BasicMeter.get(self.exp_dict["target_loss"], recursive=True, force=False).mean()
        self.scheduler.step(loss)  # update the learning rate monitor
        return {"val_loss": val_loss_meter.mean(), "val_accuracy": val_accuracy_meter.mean()}

    @torch.no_grad()
    def test_on_loader(self, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        test_loss_meter = BasicMeter.get("test_loss").reset()
        test_accuracy_meter = BasicMeter.get("test_accuracy").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, _data in enumerate(data_loader):
            batch = _data[0]
            loss, accuracy = self.val_on_batch(batch)
            test_loss_meter.update(float(loss), 1)
            test_accuracy_meter.update(float(accuracy), 1)
        return {"test_loss": test_loss_meter.mean(), "test_accuracy": test_accuracy_meter.mean()}

    def get_state_dict(self):
        """Obtains the state dict of this model including optimizer, scheduler, etc
        
        Returns:
            dict: state dict
        """
        ret = {}
        ret["optimizer"] = self.optimizer.state_dict()
        ret["model"] = self.model.state_dict()
        ret["scheduler"] = self.scheduler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        """Loads the state of the model
        
        Args:
            state_dict (dict): The state to load
        """
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])
        self.scheduler.load_state_dict(state_dict["scheduler"])

    def get_lr(self):
        ret = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            ret["current_lr_%d" % i] = float(param_group["lr"])
        return ret

    def is_end_of_training(self):
        lr = self.get_lr()["current_lr_0"]
        return lr <= (self.exp_dict["lr"] * self.exp_dict["min_lr_decay"])