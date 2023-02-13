import torch

from .pre_trained_trans import load_MLM_transformer
from .tokenizers import load_tokenizer

class PromptFinetuning(torch.nn.Module):
    '''
    CLASS DESIGNED TO BE ATTACKED ONLY, so assume batch_size 1 (and no attention mask used)
    '''
    
    def __init__(
            self, 
            trans_name:str, 
            label_words:list, 
    ):
        super().__init__()
        self.transformer = load_MLM_transformer(trans_name)
        self.tokenizer   = load_tokenizer(trans_name)
        self.label_ids   = [self.tokenizer(word).input_ids[1] for word in label_words]

        self.prompt_ids = self.tokenizer.convert_tokens_to_ids(['It', 'was', self.tokenizer.mask_token, '.'])
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):    

        # add prompt ids
        input_ids = torch.LongTensor([self.add_prompt_ids(ids.cpu().tolist()) for ids in input_ids])
        mask_positions = torch.LongTensor([ids.cpu().tolist().index(self.tokenizer.mask_token_id) for ids in input_ids])

        # encode everything and get MLM probabilities
        trans_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # select MLM probs of the masked positions, only for the label ids
        mask_pos_logits = trans_output.logits[torch.arange(input_ids.size(0)), mask_positions]
        logits = mask_pos_logits[:, tuple(self.label_ids)]

        return logits
    
    def add_prompt_ids(self, ids):
        eos_pos = ids.index(self.tokenizer.eos_token_id)
        ids = ids[:eos_pos] + self.prompt_ids + ids[eos_pos:]
        return ids

    def update_label_words(self, label_words:str):
        self.label_ids = [self.tokenizer(word).input_ids[1] for word in label_words]

    def freeze_head(self):
        for param in self.transformer.lm_head.parameters():
            param.requires_grad = False

    def predict(self, sentences, device=torch.device('cpu')):

        ml = self.tokenizer.model_max_length if self.tokenizer.model_max_length < 5000 else 512
        inputs = self.tokenizer(sentences, max_length=ml, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        # attention_mask = inputs['attention_mask']

        input_ids = input_ids.to(device)
        # attention_mask = attention_mask.to(device)
        
        return(self.forward(input_ids))