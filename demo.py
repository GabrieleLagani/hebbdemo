import os
import torch
import torch.nn as nn

from experiment import run


DATASET = 'cifar10'
WHITEN_LVL = None
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
MOMENTUM = 0.9
WDECAY=5e-4
SCHED_MILESTONES = range(30, 50, 2)
SCHED_GAMMA = 0.5
HEBB_PARAMS = {'mode': 'swta', 'w_nrm': True, 'bias': False, 'act': nn.Identity(), 'k': 50, 'alpha': 1.}
#HEBB_PARAMS = {'mode': 'hpca', 'w_nrm': False, 'bias': False, 'act': nn.Identity(), 'k': 1, 'alpha': 1.}


if __name__ == '__main__':
	run(
		exp_name=os.path.basename(__file__).rsplit('.', 1)[0],
		dataset=DATASET,
		whiten_lvl=WHITEN_LVL,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
	    lr=LR,
		momentum=MOMENTUM,
		wdecay=WDECAY,
		sched_milestones=SCHED_MILESTONES,
		sched_gamma=SCHED_GAMMA,
		hebb_params=HEBB_PARAMS,
	)

