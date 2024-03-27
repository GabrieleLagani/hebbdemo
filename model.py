import torch
import torch.nn as nn

from hebb import HebbianConv2d

default_hebb_params = {'mode': HebbianConv2d.MODE_SWTA, 'w_nrm': True, 'k': 50, 'act': nn.Identity(), 'alpha': 0.}

class Net(nn.Module):
	def __init__(self, hebb_params=None):
		super().__init__()
		
		if hebb_params is None: hebb_params = default_hebb_params
		
		# A single convolutional layer
		self.conv1 = HebbianConv2d(3, 64, 5, 2, **hebb_params)
		self.bn1 = nn.BatchNorm2d(64, affine=False)
		
		# Aggregation stage
		self.pool = nn.AdaptiveAvgPool2d(1)
		
		# Final fully-connected 2-layer classifier
		hidden_shape = self.get_hidden_shape()
		self.fc2 = HebbianConv2d(hidden_shape[0], 256, (hidden_shape[1], hidden_shape[2]), **hebb_params)
		self.bn2 = nn.BatchNorm2d(256, affine=False)
		self.fc3 = nn.Linear(256, 10)
	
	def get_hidden_shape(self):
		self.eval()
		with torch.no_grad(): out = self.forward_features(torch.ones([1, 3, 32, 32], dtype=torch.float32)).shape[1:]
		return out
	
	def forward_features(self, x):
		x = self.bn1(torch.relu(self.conv1(x)))
		x = self.pool(x)
		return x
	
	def forward(self, x):
		x = self.forward_features(x)
		x = self.bn2(torch.relu(self.fc2(x)))
		x = self.fc3(torch.dropout(x.reshape(x.shape[0], x.shape[1]), p=0.5, train=self.training))
		return x
	