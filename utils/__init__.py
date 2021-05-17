from utils.util import get_prune_params,get_prune_summary,log_obj
from utils.util import create_model,copy_model
from utils.util import evaluate,fevaluate,train,ftrain
from utils.util import prune_fixed_amount,fprune_fixed_amount,average_weights_masks

from utils.globalpruner import GlobalPruner
from utils.globalpruner import super_prune
from utils.globalpruner import globalPrunerStructured