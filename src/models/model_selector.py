from .base_models import SequenceClassifier
from .prompt_models import PromptFinetuning

import torch

def select_model(model_name='bert-base-uncased', model_path=None, pretrained=True, num_labels=2, prompt_finetune=False):
    if prompt_finetune:
        model = PromptFinetuning(model_name, ['terrible', 'great'])
    else:
        model =  SequenceClassifier(model_name=model_name, pretrained=pretrained, num_labels=num_labels)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    return model