import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
# from torch_geometric.nn import GCNConv, Linear, to_hetero

# from args import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')


class GraphConvSparse(nn.Module):
	"""
	Graph convolution layer for sparse inputs.
	"""
	def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.weight = glorot_init(input_dim, output_dim) 
		self.activation = activation
		self.dropout = 0
	
	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.weight)

	def forward(self, inputs, adj):
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

class EdgeDecoder(nn.Module):
	def __init__(self, dropout):
		super(EdgeDecoder, self).__init__()
		self.dropout = dropout
		self.a = nn.Parameter(torch.FloatTensor(1))
		self.b = nn.Parameter(torch.FloatTensor(1))
		# nn.init.constant_(self.a, 0.8)
		# nn.init.constant_(self.b, 7)
		# nn.init.constant_(self.a, 9)
		# nn.init.constant_(self.b, 16)
		nn.init.constant_(self.a, 1.65)
		nn.init.constant_(self.b, 10)
		
	
	def forward(self, z_c, z_t, method):
		"""
		:param z_c: torch.Tensor, latent tensor for company
		:param z_t: torch.Tensor, latent tensor for term
		:param method: str, method to combine latent tensors
		"""
		z_c = F.dropout(z_c, self.dropout, training=self.training)
		z_t = F.dropout(z_t, self.dropout, training=self.training)
		if method == 'dot':
			A_pred = torch.sigmoid(torch.mm(z_c, z_c.t()))
			B_pred = torch.sigmoid(torch.mm(z_c, z_t.t()))
		elif method == 'ed':
			eps = 1e-8
			distance_cc = torch.cdist(z_c,z_c, p=2)
			x = 1/(distance_cc+eps)
			A_pred = torch.sigmoid((x/self.a)-self.b)
			distance_ct = torch.cdist(z_c,z_t, p=2)
			x = 1/(distance_ct+eps)
			B_pred = torch.sigmoid((x/self.a)-self.b)
		return A_pred, B_pred

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			+ ' (z_company: 2->4555)' \
			+ ' (z_term: 2->4555)' \
		+ ')'
	


class MLP(nn.Module):
	def __init__(self, bipartite_dim, hidden1_dim, hidden2_dim, dropout=0.0):
		super(MLP,self).__init__()
		self.bipartite_dim = bipartite_dim
		self.hidden1_dim = hidden1_dim
		self.hidden2_dim = hidden2_dim
		self.l1 = nn.Linear(bipartite_dim, hidden1_dim)
		self.l21 = nn.Linear(hidden1_dim, hidden2_dim) # for mu layer
		self.l22 = nn.Linear(hidden1_dim, hidden2_dim) # for sigma layer
		self.dropout = dropout
	
	def reparameterize(self, mu, logvar):
		if self.training:
			std = torch.exp(logvar)
			eps = torch.randn_like(std)
			return eps.mul(std).add_(mu)
		else:
			return mu
	
	def forward(self, bi_network):
		h1 = self.l1(bi_network)
		h1 = F.relu(h1)
		self.mu = self.l21(h1)
		self.sigma = self.l22(h1)
		z = self.reparameterize(self.mu, self.sigma)
		return z, self.mu, self.sigma

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			+ str(self.bipartite_dim) + ' ->' \
			+ str(self.hidden1_dim) + ' ->' \
			+ str(self.hidden2_dim) + ')'



class HeteroVGAE(nn.Module):
	"""
	Variational Graph Auto-Encoder (VGAE) model for heterogeneous graphs.
	"""
	def __init__(self, graph_dim, bipartite_dim, hidden1_dim, hidden2_dim, dropout=0.0):
		super(HeteroVGAE,self).__init__()
		self.base_gcn = GraphConvSparse(graph_dim, hidden1_dim)
		self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x:x)
		self.mlp = MLP(bipartite_dim, hidden1_dim, hidden2_dim, dropout)
		self.decoder = EdgeDecoder(dropout)
		self.dropout = dropout


	def encode(self, X, adj):
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
	
	
	def loss_function(self, norm, norm_bi, adj_label, bi_network_label, A_pred, bipartite_pred, weight_tensor, weight_tensor_bi):
		# Reconstruction loss
		eps = 1e-8

		recon_loss1 = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1).to(device), weight=weight_tensor.to(device))
		recon_loss2 = norm_bi * F.binary_cross_entropy(bipartite_pred.view(-1), bi_network_label.to_dense().view(-1).to(device), weight=weight_tensor_bi.to(device))

		# KL divergence loss
		KLD1 = -0.5/A_pred.size(0) * torch.sum(1 + self.logstd - self.mean.pow(2) - self.logstd.exp())
		KLD2 = -0.5/bipartite_pred.size(0) * torch.sum(1 + self.sigma - self.mu.pow(2) - self.sigma.exp())
		
		loss = recon_loss1 + recon_loss2 + KLD1 + KLD2
		
		return loss
	

	def predict(self, X, adj, bi_network):
		mean, logstd = self.encode(X, adj)
		self.z_c = self.reparameterize(mean, logstd)
		self.z_t, self.mu, self.sigma = self.mlp(bi_network)
		# A_pred, bipartite_pred = self.decoder(self.z_c, self.z_t, method='dot')
	
		# return self.z_c, self.z_t, A_pred, bipartite_pred
		return self.z_c, self.z_t
		

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
		self.z_t, self.mu, self.sigma = self.mlp(bi_network)
		A_pred, bipartite_pred = self.decoder(self.z_c, self.z_t, method='ed')
	
		return self.z_c, self.z_t, A_pred, bipartite_pred
	

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)


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