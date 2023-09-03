import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class PhoBERTChatBot(nn.Module):    
    def __init__(self, model_name, output_size):
        super(PhoBERTChatBot, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(4*self.bert.config.hidden_size, output_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        cls_output = torch.cat((outputs[2][-1][:,0, ...], outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
        outputs = self.fc(cls_output)
        return outputs


class PhoBERTCustom(nn.Module):
    def __init__(self, model_name, output_size):
        super(PhoBERTCustom, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_size)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)
        
    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        x = self.dropout(output)
        x = self.fc(x)
        return x


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l1(x))
        out = self.l3(out)
        return out
