import torch

from .pre_trained_trans import load_MLM_transformer
from .tokenizers import load_tokenizer

class PromptFinetuning(torch.nn.Module):
    def __init__(
            self, 
            trans_name:str, 
            label_words:list, 
    ):
        super().__init__()
        self.transformer = load_MLM_transformer(trans_name)
        self.tokenizer   = load_tokenizer(trans_name)
        self.label_ids   = [self.tokenizer(word).input_ids[1] for word in label_words]
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        mask_positions=None,
    ):
        # encode everything and get MLM probabilities
        trans_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # select MLM probs of the masked positions, only for the label ids
        mask_pos_logits = trans_output.logits[torch.arange(input_ids.size(0)), mask_positions]
        logits = mask_pos_logits[:, tuple(self.label_ids)]

        return logits

    def update_label_words(self, label_words:str):
        self.label_ids = [self.tokenizer(word).input_ids[1] for word in label_words]

    def freeze_head(self):
        for param in self.transformer.lm_head.parameters():
            param.requires_grad = False

    def predict(self, sentences, output_attentions=False, output_hidden_states=False, return_dict=False, device=torch.device('cpu')):
        ml = self.tokenizer.model_max_length if self.tokenizer.model_max_length < 5000 else 512
        inputs = self.tokenizer(sentences, padding=True, max_length=ml, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        return(self.model(input_ids, attention_mask, output_attentions=output_attentions,
                 output_hidden_states=output_hidden_states, return_dict=return_dict))