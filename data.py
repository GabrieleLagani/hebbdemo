import os

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

import params as P


# Transformations to be applied to the training data
T_trn = T.Compose([
	T.Resize(32), # Resize shortest size of the image to a fixed size.
	T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=30/360)], p=0.3), # Random color jittering
	T.RandomApply([T.ColorJitter(saturation=1.)], p=0.1), # Randomly transform image saturation from nothing to full grayscale
	T.RandomHorizontalFlip(), # Randomly flip image horizontally.
	T.RandomVerticalFlip(), # Randomly flip image vertically.
	T.RandomApply([T.Pad(16, padding_mode='reflect'), T.RandomRotation(10), T.CenterCrop(32)], p=0.3), # Random rotation
	T.RandomApply([T.RandomCrop(32, padding=8, pad_if_needed=True, padding_mode='reflect')], p=0.3), # Random translation and final cropping with fixed size
	T.ToTensor(),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Transformations to be applied to the test data
T_tst = T.Compose([
	T.Resize(32), # Resize shortest size of the image to a fixed size.
	T.CenterCrop(32), # Center crop of fixed size
	T.ToTensor(),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_data(dataset='cifar10', root='datasets', batch_size=32, num_workers=0, whiten_lvl=None):
	trn_set, tst_set = None, None
	if dataset == 'cifar10':
		trn_set = DataLoader(CIFAR10(root=os.path.join(root, dataset), train=True, download=True, transform=T_trn), batch_size=batch_size, shuffle=True, num_workers=P.NUM_WORKERS)
		tst_set = DataLoader(CIFAR10(root=os.path.join(root, dataset), train=False, download=True, transform=T_tst), batch_size=batch_size, shuffle=False, num_workers=P.NUM_WORKERS)
	else: raise NotImplementedError("Dataset {} not supported.".format(dataset))
	
	zca = None
	if whiten_lvl is not None:
		raise NotImplemented("Whitening not implemented.")
	
	return trn_set, tst_set, zca

def whiten(inputs, zca):
	raise NotImplemented("Whitening not implemented.")

