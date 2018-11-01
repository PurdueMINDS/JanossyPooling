import torch
import torch.nn as nn
from torch.nn import init
from torchvision.models import inception_v3


class TextModels(nn.Module):

	def __init__(self, vocab_size, input_dim, model, num_layers, num_neurons, janossy_k, device):
		"""Create a model based on the request"""
		super(TextModels, self).__init__()
		self.num_layers = num_layers
		self.num_neurons = num_neurons
		self.vocab_size = vocab_size
		self.input_dim = int(input_dim/janossy_k)
		self.input_dim_mod = self.input_dim * janossy_k	
		self.device = device
		self.model_name = model

		# Define the loss function
		self.loss_func = nn.L1Loss()
	
		# Embedding Layer which is non trainable
		self.emb = nn.Embedding(self.vocab_size, self.input_dim)
		init.uniform_(self.emb.weight,a=-0.5,b=0.5)
		self.emb.weight.requires_grad = False
		
		# Create the model here based on input
		if self.model_name == 'lstm':
			self.model = nn.LSTM(self.input_dim_mod, 50, batch_first=True)
			self.model_activation = None
			self.model_out_shape = 50
		elif self.model_name == 'gru':
			self.model = nn.GRU(self.input_dim_mod, 80, batch_first=True)
			self.model_activation = None
			self.model_out_shape = 80
	
		else:
			self.model = nn.Linear(self.input_dim_mod, 30)
			self.model_activation = nn.Tanh()
			self.model_out_shape = 30
			init.xavier_uniform_(self.model.weight)
			self.model.bias.data.fill_(0)	
		
		# Multiple Hidden Layers based on input
		# Neurons in Hidden Layer based on input
		self.rho_mlp_linear = []
		for i in range(num_layers):
			if i == 0:
				self.rho_mlp_linear.append(nn.Linear(self.model_out_shape, num_neurons))
			else:
				self.rho_mlp_linear.append(nn.Linear(num_neurons, num_neurons))
			init.xavier_uniform_(self.rho_mlp_linear[-1].weight)
			self.rho_mlp_linear[-1].bias.data.fill_(0)
			self.rho_mlp_linear.append(nn.Tanh())
		if self.num_layers == 0:
			self.final_layer = nn.Linear(self.model_out_shape, 1)
		else:	
			for layer_num in range(len(self.rho_mlp_linear)):
				self.add_module("hidden_"+str(layer_num),self.rho_mlp_linear[layer_num])
			self.final_layer = nn.Linear(self.num_neurons, 1)
		init.xavier_uniform_(self.final_layer.weight)
		self.final_layer.bias.data.fill_(0)

	def forward(self, input_tensor):
		"""Lookup the tensor and then continue with feedforward"""
		# Input as a long tensor
		emb_output = self.emb(input_tensor)
		emb_shape = emb_output.shape
		emb_output = emb_output.view(emb_shape[0], emb_shape[1], -1)
		# Feed the obtained embedding to the Janossy Layer
		if self.model_activation is not None:
			model_out = self.model(emb_output)
			model_out = self.model_activation(model_out)
		else:
			model_out, _ = self.model(emb_output)
			model_out = model_out[:, -1, :]  # Just the final state
		if self.model_name in ['lstm','gru']:
			rho_out = model_out
		else:
			summer_out = torch.sum(model_out, dim=1).to(self.device)
			rho_out = summer_out
		for layer_num in range(len(self.rho_mlp_linear)):
			rho_out = getattr(self,"hidden_"+str(layer_num))(rho_out)
		final_output = self.final_layer(rho_out)
		return final_output	

	def loss(self, input_tensor, output_tensor):
		"""Loss Computations"""
		predicted_output = self.forward(input_tensor)
		return self.loss_func(predicted_output, output_tensor)

