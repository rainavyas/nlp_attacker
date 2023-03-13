from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

class Constraint():
    '''
        Check constraints as per different attack methods
    '''
    def __init__(self):
        self.mapper = {
            'bae'   :   '_bae_constraints'
        }

    def check_constraint(self, orig, adv, attack_method):
        constraints = getattr(self, self.mapper[attack_method])()
        for c in constraints:
            if not c._check_constraint(adv, orig):
                return False
        return True
    
    @staticmethod
    def _bae_constraints():
        # Difficult to include pre-transformation constraints
        # constraints = [RepeatModification(), StopwordModification()]
        # constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        constraints = []
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        return constraints