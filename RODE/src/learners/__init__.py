from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .rode_learner import RODELearner
from .EWM_learner import EWMLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["rode_learner"] = RODELearner
REGISTRY["EWM_learner"] = EWMLearner
