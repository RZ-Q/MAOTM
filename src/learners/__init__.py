from .q_learner import QLearner
from .policy_infer_q_learner import PIQLearner
from .mbom_learner import MBOMLearner
from .rbom_learner import RBOMLearner
# from .rbom_learner_new import RBOMLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["policy_infer_q_learner"] = PIQLearner
REGISTRY['mbom_learner'] = MBOMLearner
REGISTRY['rbom_learner'] = RBOMLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner

