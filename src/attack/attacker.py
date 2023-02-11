import textattack
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from tqdm import tqdm
import torch

from .model_wrapper import PyTorchModelWrapper

class Attacker():
    
    @classmethod
    def attack_all(cls, data, model, method='textfooler'):
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
        attack = cls._construct_attack(model_wrapper, method)
        
        # Add orig predicted class
        print('Adding original predicted class')
        cls._pred_class(data, model)

        # attack
        print("Attacking")
        for d in tqdm(data):
            att_text = cls.attack(d['text'], d['label'], attack)
            d['att_text'] = att_text
        
        # Add attacked predicted class
        print('Adding attacked predicted class')
        cls._pred_class(data, model, text_key='att_text', lab_key='att_pred_label')

        return data
            


    @staticmethod
    def _construct_attack(model_wrapper, method):
        if method == 'textfooler':
            attack = TextFoolerJin2019.build(model_wrapper)
        return attack

    @staticmethod
    def attack(sentence, y, attack):
        attack_result = attack.attack(sentence, y)
        att_text = attack_result.perturbed_text()
        return att_text

    
    def _pred_class(data, model, text_key='text', lab_key='pred_label'):
        '''Add predictions to data'''
        for d in tqdm(data):
            with torch.no_grad():
                logits = model.predict([d[text_key]])[0].squeeze()
                y = torch.argmax(logits).detach().cpu().item()
                d[lab_key] = y