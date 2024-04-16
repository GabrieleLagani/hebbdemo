import torch

AVAILABLE_DEVICES = ['cpu']
if torch.cuda.is_available(): AVAILABLE_DEVICES += ['cuda:{}'.format(d) for d in range(torch.cuda.device_count())]
DEFAULT_DEVICE = 'cuda:0' #'cpu'
NUM_WORKERS = 4
DIST_BINS = 20

