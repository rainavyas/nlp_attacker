from .model_selector import select_model
from ..training.trainer import Trainer

class Ensemble():
    def __init__(self, model_names, model_paths, device, num_labels=2, prompt_finetune=False):
        self.models = []
        for mname, mpath in zip(model_names, model_paths):
            model = select_model(mname, mpath, num_labels=num_labels, prompt_finetune=prompt_finetune)
            model.to(device)
            self.models.append(model)

    def eval(self, dl, criterion, device):
        '''
        Evaluate Ensemble predictions
        Returns list of accuracies
        '''
        accs = []
        for m in self.models:
            acc = Trainer.eval(dl, m, criterion, device)
            accs.append(acc)
        return accs