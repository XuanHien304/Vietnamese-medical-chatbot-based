import torch
from torch import nn as nn
import json
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoModel, AutoTokenizer, AdamW
from vncorenlp import VnCoreNLP

rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
BATCH_SIZE = 4

class ChatBotDataset(Dataset):
    def __init__(self, data_dir, tokenizer, batch_size):
        super(ChatBotDataset, self).__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        with open(self.data_dir, 'r', encoding='utf-8') as c:
            contents = json.load(c)
        
        tags = []
        X = []
        y = []

        for content in contents['intents']:
            tag = content['tag']
            for pattern in content['patterns']:
                X.append(pattern)
                tags.append(tag)

        tags_set = sorted(set(tags))
        for tag in tags:
            label = tags_set.index(tag)
            y.append(label)

        encode_dict = self.tokenizer.encode_plus(X,
                                                max_length=64,
                                                padding='max_length',
                                                truncation=True,
                                                return_attention_mask=True,
                                                return_token_type_ids=False,
                                                return_tensors='pt'
                                                )
        input_ids = encode_dict['input_ids'][0]
        attention_mask = encode_dict['attention_mask'][0]
        y_train = torch.LongTensor(y)
        return input_ids, attention_mask, y_train


def data_loader(data_dir, tokenizer, batch_size=BATCH_SIZE):
    with open(data_dir, 'r', encoding='utf-8') as c:
        contents = json.load(c)

    tags = []
    X = []
    y = []

    for content in contents['intents']:
        tag = content['tag']
        for pattern in content['patterns']:
            X.append(pattern)
            tags.append(tag)

    tags_set = sorted(set(tags))

    for tag in tags:
        label = tags_set.index(tag)
        y.append(label)

    X = [' '.join(rdrsegmenter.tokenize(x)[0]) for x in X]
    token_train = {}
    token_train = tokenizer.batch_encode_plus(  X,
                                                max_length=64,
                                                padding='max_length',
                                                truncation=True,
                                                return_attention_mask=True,
                                                return_token_type_ids=False,
                                                return_tensors='pt')

    X_train_mask = token_train['attention_mask']
    X_train = token_train['input_ids']
    y_train = torch.LongTensor(y)

    dataset = TensorDataset(X_train, X_train_mask, y_train)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    return data_loader

if __name__ == '__main__':
    train_dir = './intent_train.json'
    val_dir = './intent_val.json'
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    train_data_loader = data_loader(train_dir, tokenizer, batch_size=BATCH_SIZE)
    val_data_loader = data_loader(val_dir, tokenizer, batch_size=BATCH_SIZE)
    print()