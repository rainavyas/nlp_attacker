from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
# from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from .use import UniversalSentenceEncoder
from textattack.shared.attacked_text import AttackedText

from textattack.shared.utils import LazyLoader

# hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")

class Constraint():
    '''
        Check constraints as per different attack methods
    '''
    def __init__(self, attack_method):
        mapper = {
            'bae'   :   '_bae_constraints'
        }

        self.constraints = getattr(self, mapper[attack_method])()

    def check_constraint(self, orig, adv):
        orig = AttackedText(orig)
        adv = AttackedText(adv)

       # check semantic constraint for central window only (set a large window)
        orig.attack_attrs["newly_modified_indices"] = [0]
        adv.attack_attrs["newly_modified_indices"] = [0]

        for c in self.constraints:
            if not c._check_constraint(adv, orig):
                return False
        return True
    
    def _bae_constraints(self):
        # Do NOT include pre-transformation constraints
        # constraints = [RepeatModification(), StopwordModification()]
        # constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            # window_size=15, # original window size used by BAE - applied at every change index
            window_size=100,
            # skip_text_shorter_than_window=True,
            skip_text_shorter_than_window=False,
        )
        # have to call encode once to create self.model
        return [use_constraint]