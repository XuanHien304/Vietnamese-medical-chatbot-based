import numpy as np
import random
import torch
from torch import nn as nn
import json
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, AdamW
from src.dataset import data_loader
from src.model import PhoBERTChatBot, PhoBERTCustom
from src.trainer import Trainer

CUDA_LAUNCH_BLOCKING=1
SEED = 3004

def set_seeds(seed):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seeds(seed=SEED)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    EPOCHS = 100
    train_dir = './data/intent_train.json'
    val_dir = './data/intent_val.json'
    LEARNING_RATE = 1e-4
    PATIENCE = 20

    model = PhoBERTChatBot('vinai/phobert-base', 8)
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    loss_criteria = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3)

    trainer = Trainer(model=model, device=device, optimizer=optimizer,
                    loss_fn=loss_criteria, scheduler=scheduler
                        )
    train_data_loader, val_data_loader = trainer.set_up_training_data(train_dir, val_dir, tokenizer)
    best_model = trainer.train(EPOCHS, PATIENCE, train_data_loader, val_data_loader)
    return best_model
