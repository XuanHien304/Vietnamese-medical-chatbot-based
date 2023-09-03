import numpy as np
import torch
from tqdm import tqdm
from src.dataset import data_loader

BATCH_SIZE = 8
SAVE_PATH_MODEL = '/weight'
class Trainer():
    def __init__(self, model, device, optimizer, loss_fn, scheduler=None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def train_epoch(self, train_data_loader):
        self.model.train()
        train_loss = 0
        correct = 0
        pbar = tqdm(enumerate(train_data_loader), total = len(train_data_loader))
        for i, batch in pbar:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            self.model.zero_grad()
            output = self.model(b_input_ids, b_input_mask)
            loss = self.loss_fn(output, b_labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(output, dim=1)
            correct += torch.sum(preds == b_labels)
        avg_loss = train_loss/len(train_data_loader)
        acc = correct.double()/len(train_data_loader.dataset)
        return avg_loss, acc

    def eval_epoch(self, val_data_loader):
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.inference_mode():
            pbar = tqdm(enumerate(val_data_loader), total = len(val_data_loader))
            for i, batch in pbar:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                output = self.model(b_input_ids, b_input_mask)
                loss = self.loss_fn(output, b_labels)
                val_loss += loss.item()

                _, preds = torch.max(output, dim=1)
                correct += torch.sum(preds == b_labels)
        avg_loss = val_loss/len(val_data_loader)
        acc = correct.double()/len(val_data_loader.dataset)
        return avg_loss, acc

    def train(self, epochs, patience, train_loader, val_loader):
        best_val_loss = np.inf
        for epoch in range(epochs):

            avg_train_loss, train_acc = self.train_epoch(train_data_loader=train_loader)
            avg_val_loss, val_acc = self.eval_epoch(val_data_loader=val_loader)
            if self.scheduler:    
                self.scheduler.step(avg_train_loss)
            
            # Ealry stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = self.model
                torch.save(self.model.state_dict(), 'saved_weights.pth')
                _patience = patience
            else:
                _patience -= 1
            if not _patience:
                print('Stopping early')
                break

            print(
                f"Epoch: {epoch} | "
                f"train_loss: {avg_train_loss:.5f}, "
                f"train_acc: {train_acc:.5f}, "
                f"val_loss: {avg_val_loss:.5f}, "
                f"val_acc: {val_acc:.5f}, "
                f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                f"_patience: {_patience}"
            )
        return best_model

    def set_up_training_data(self, train_dir, val_dir, tokenizer):
        print('----- Setting up data ... -----')
        train_data_loader = data_loader(train_dir, tokenizer, batch_size=BATCH_SIZE)
        val_data_loader = data_loader(val_dir, tokenizer, batch_size=BATCH_SIZE)
        return train_data_loader, val_data_loader
