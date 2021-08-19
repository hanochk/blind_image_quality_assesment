# from lib.include import *
# from lib.utility.draw import *
# from lib.utility.file import *
# from lib.net.rate import *
import torch
import numpy as np
import random
import os
import time
#---------------------------------------------------------------------------------
COMMON_STRING ='@%s:  \n' % os.path.basename(__file__)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.deterministic = True
    return seed

if 1:
    seed = seed_all(int(time.time())) #35202   #35202  #123  #

    COMMON_STRING += '\tpytorch environment:\n'
    COMMON_STRING += '\t\ttorch.__version__                  = %s\n'%torch.__version__
    COMMON_STRING += '\t\ttorch.version.cuda                 = %s\n'%torch.version.cuda
    COMMON_STRING += '\t\ttorch.backends.cudnn.version()     = %s\n'%torch.backends.cudnn.version()
    COMMON_STRING += '\t\ttorch.backends.cudnn.enabled       = %s\n'%torch.backends.cudnn.enabled
    COMMON_STRING += '\t\ttorch.backends.cudnn.deterministic = %s\n'%torch.backends.cudnn.deterministic
    COMMON_STRING += '\t\ttorch.cuda.device_count()          = %d\n'%torch.cuda.device_count()

    try:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\'] = %s\n'%os.environ['CUDA_VISIBLE_DEVICES']
    except Exception:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\'] = None\n'

    COMMON_STRING += '\t\tseed = %d\n'%seed



COMMON_STRING += '\n'

#---------------------------------------------------------------------------------
## useful : http://forums.fast.ai/t/model-visualization/12365/2


if __name__ == '__main__':
    print (COMMON_STRING)
