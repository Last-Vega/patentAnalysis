import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from .args import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')


class GraphConvSparse(nn.Module):
	"""
	Graph convolution layer for sparse inputs.
	"""
	def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
		"""
		:param input_dim: int, input dimension
		:param output_dim: int, output dimension
		:param activation: function, activation function
		:param kwargs: dict, other named arguments
		"""
		super(GraphConvSparse, self).__init__(**kwargs)
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.weight = glorot_init(input_dim, output_dim) 
		self.activation = activation
		self.dropout = 0
	
	def reset_parameters(self):
		"""
		Reinitialize learnable parameters.
		"""
		torch.nn.init.xavier_uniform_(self.weight)

	def forward(self, inputs, adj):
		"""
		:param inputs: torch.sparse.FloatTensor, input sparse tensor
		:param adj: torch.sparse.FloatTensor, adjacency matrix
		:return: torch.Tensor, output tensor
		"""
		x = inputs
		x = F.dropout(x, self.dropout, training=self.training)
		x = torch.mm(x, self.weight)
		x = torch.mm(adj, x)
		outputs = self.activation(x)
		return outputs
	
	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			+ str(self.input_dim) + ' ->' \
			+ str(self.output_dim) + ')'

class VGAE(nn.Module):
	def __init__(self, input_dim, hidden1_dim, hidden2_dim, dropout=0.0):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(input_dim, hidden1_dim)
		self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x:x)
		self.a = nn.Parameter(torch.FloatTensor(1))
		self.b = nn.Parameter(torch.FloatTensor(1))
		nn.init.constant_(self.a, 0.8)
		nn.init.constant_(self.b, 7)
		self.decoder = EdgeDecoder(dropout)


	def encode(self, X, adj):
		hidden = self.base_gcn(X, adj)
		self.mean = self.gcn_mean(hidden, adj)
		self.logstd = self.gcn_logstddev(hidden, adj)
		self.mu = self.mean
		self.logvar = self.logstd
		return self.mu, self.logvar
	
	def reparameterize(self, mu, logvar):
		if self.training:
			std = torch.exp(logvar)
			eps = torch.randn_like(std)
			return eps.mul(std).add_(mu)
		else:
			return mu
	
	def predict_z(self, X, adj):
		mu, logvar = self.encode(X, adj)
		z = self.reparameterize(mu, logvar)
		return z

	def forward(self, X, adj):
		mu, logvar = self.encode(X, adj)
		self.z = self.reparameterize(mu, logvar)
		A_pred = self.decoder(self.z, self.a, self.b, method='dot')
		return self.z, mu, logvar, A_pred


class EdgeDecoder(nn.Module):
	"""
	Edge decoder model.
	"""
	def __init__(self, dropout):
		"""
		:param dropout: float, dropout rate
		"""
		super(EdgeDecoder, self).__init__()
		self.dropout = dropout
		
	
	def forward(self, z, a, b, method):
		"""
		指定された方法に基づいてエッジの生成確率を計算する
		:param z: torch.Tensor, latent space
		:param a: float, parameter a
		:param b: float, parameter b
		:param method: str, method to compute adjacency matrix
		:return: torch.Tensor, adjacency matrix
		"""
		z = F.dropout(z, self.dropout, training=self.training)

		if method == 'dot':
			A_pred = torch.sigmoid(torch.mm(z, z.t()))
		elif method == 'ed':
			eps = 1e-8
			distance = torch.cdist(z,z, p=2)
			x = 1/(distance+eps)
			A_pred = torch.sigmoid((x/a)-b)
		
		return A_pred
		


class HeteroVGAE(nn.Module):
	"""
	Variational Graph Auto-Encoder (VGAE) model for heterogeneous graphs.
	"""
	def __init__(self, graph_dim, bipartite_dim, dropout=0.0):
		"""
		:param graph_dim: int, dimension of the graph
		:param bipartite_dim: int, dimension of the bipartite graph
		:param dropout: float, dropout rate
		"""
		super(HeteroVGAE,self).__init__()
		self.base_gcn = GraphConvSparse(graph_dim, hidden1_dim)
		self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x:x)
		self.a = nn.Parameter(torch.FloatTensor(1))
		self.b = nn.Parameter(torch.FloatTensor(1))
		nn.init.constant_(self.a, 0.8)
		nn.init.constant_(self.b, 7)
		self.decoder = EdgeDecoder(dropout)
		# self.weight = glorot_init(bipartite_dim, hidden2_dim)
		self.l1 = nn.Linear(bipartite_dim, hidden1_dim)
		self.l21 = nn.Linear(hidden1_dim, hidden2_dim) # for mu layer
		self.l22 = nn.Linear(hidden1_dim, hidden2_dim) # for sigma layer


	def encode(self, X, adj):
		"""
		:param inputs: torch.sparse.FloatTensor, input sparse tensor
		:param adj: torch.sparse.FloatTensor, adjacency matrix
		:return: torch.Tensor, output tensor
		"""
		hidden = self.base_gcn(X, adj)
		self.mean = self.gcn_mean(hidden, adj)
		self.logstd = self.gcn_logstddev(hidden, adj)
	
		return self.mean, self.logstd
	
	def reparameterize(self, mu, logvar):
		"""
		:param mu: torch.Tensor, mean of the latent distribution
		:param logvar: torch.Tensor, log variance of the latent distribution
		:return: torch.Tensor, latent variable
		"""
		if self.training:
			std = torch.exp(logvar)
			eps = torch.randn_like(std)
			return eps.mul(std).add_(mu)
		else:
			return mu
	
	
	def encoder_with_MLP(self, bi_network):
		"""
		:param bi_network: torch.Tensor, bipartite network
		:return: torch.Tensor, latent variable
		"""
		h1 = self.l1(bi_network)
		h1 = F.relu(h1)
		self.mu = self.l21(h1)
		self.sigma = self.l22(h1)
		
		z = self.reparameterize(self.mu, self.sigma)
		self.Z_t = z
		return z, self.mu, self.sigma
	
	def bipartite_decode(self, z_c, z_t, a, b, method):
		"""
		:param z_c: torch.Tensor, latent variable of the graph
		:param z_t: torch.Tensor, latent variable of the bipartite graph
		:param a: torch.Tensor, trainable parameter
		:param b: torch.Tensor, trainable parameter
		:param method: str, method to calculate the edge weights
		:return: torch.Tensor, edge weights
		"""
		if method=='dot':
			A_pred = torch.sigmoid(torch.mm(z_c, z_t.t()))
		elif method=='ed':
			eps = 1e-8
			distance = torch.cdist(z_c,z_t, p=2)
			x = 1/(distance+eps)
			A_pred = torch.sigmoid((x/a)-b)
		return A_pred
	
	def predict_z(self, X, adj, bi_network):
		"""
		潜在変数を予測する
		:param X: torch.sparse.FloatTensor, input sparse tensor
		:param adj: torch.sparse.FloatTensor, adjacency matrix
		:param bi_network: torch.Tensor, bipartite network
		:return: torch.Tensor, latent variable
		"""
		mean, logstd = self.encode(X, adj)
		z_c = self.reparameterize(mean, logstd)
		z_t, mu, sigma = self.encoder_with_MLP(bi_network)
		return z_c, z_t, mean, logstd, mu, sigma

	def forward(self, X, adj, bi_network):
		"""
		グラフの埋め込みを計算しエッジの生成確率を予測する
		:param X: torch.sparse.FloatTensor, input sparse tensor
		:param adj: torch.sparse.FloatTensor, adjacency matrix
		:param bi_network: torch.Tensor, bipartite network
		:return: torch.Tensor, latent variable
		"""
		mean, logstd = self.encode(X, adj)
		self.z_c = self.reparameterize(mean, logstd)
		A_pred = self.decoder(self.z_c, self.a, self.b, method='ed')
		self.z_t, mu, sigma = self.encoder_with_MLP(bi_network)
		bipartite_pred = self.bipartite_decode(self.z_c, self.z_t, self.a, self.b, method='ed')
	
		return self.z_c, self.z_t, mean, logstd, mu, sigma, A_pred, bipartite_pred
	
def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def norm_distance_decode(Z, a, b):
	eps = 1e-8
	distance = torch.cdist(Z,Z, p=2)
	x = 1/(distance+eps)
	A_pred = torch.sigmoid((x/a)-b)
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)
