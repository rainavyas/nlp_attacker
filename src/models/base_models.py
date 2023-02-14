from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch.nn as nn
import torch

import torch
import torch.nn as nn

from types import SimpleNamespace
from transformers import logging

from .pre_trained_trans import load_transformer
from .tokenizers import load_tokenizer

class SequenceClassifierVyas(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2, pretrained=True):
        super().__init__()
        self.model_name = model_name
        if pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels) # returns config and not pretrained weights 
            self.model = AutoModelForSequenceClassification.from_config(config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)[0]
    
    def predict(self, sentences, output_attentions=False, output_hidden_states=False, return_dict=False, device=torch.device('cpu')):
        ml = self.tokenizer.model_max_length if self.tokenizer.model_max_length < 5000 else 512
        inputs = self.tokenizer(sentences, padding=True, max_length=ml, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        return(self.model(input_ids, attention_mask, output_attentions=output_attentions,
                 output_hidden_states=output_hidden_states, return_dict=return_dict))


class TransformerModel(torch.nn.Module):
    """basic transformer model for multi-class classification""" 
    def __init__(
        self, 
        trans_name:str, 
        num_classes:int=2
    ):
        super().__init__()
        self.transformer = load_transformer(trans_name)
        h_size = self.transformer.config.hidden_size
        self.output_head = nn.Linear(h_size, num_classes)
        self.tokenizer   = load_tokenizer(trans_name)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_positions=None,
    ):
        
        # get transformer hidden representations
        trans_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # get CLS hidden vector and convert to logits through classifier
        H = trans_output.last_hidden_state  #[bsz, L, 768] 
        h = H[:, 0]                         #[bsz, 768] 
        logits = self.output_head(h)             #[bsz, C] 
        return logits

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = True
    
    def freeze_classifier_bias(self):
        self.output_head.bias.requires_grad = False

    def predict(self, sentences, device=torch.device('cpu')):

        ml = self.tokenizer.model_max_length if self.tokenizer.model_max_length < 5000 else 512
        inputs = self.tokenizer(sentences, max_length=ml, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        return(self.forward(input_ids, attention_mask))



        