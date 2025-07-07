import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_scatter import scatter_mean
from torch_geometric.nn import TransformerConv

from egnn_clean import EGNN
from data import *

def split_batch(x,batchid):
    x =  x.unsqueeze(0)
    unique_batch_ids = torch.unique(batchid)
    batchx = []
    for batch_id in unique_batch_ids:
        batch_indices = torch.nonzero(batchid == batch_id).squeeze()
        batchx.append(x[:,batch_indices])
    return batchx

class GNNLayer(nn.Module):
    """
    define GNN layer for subsequent computations
    """
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(
            in_channels=num_hidden, 
            out_channels=int(num_hidden / num_heads), 
            heads=num_heads, dropout = dropout, 
            edge_dim = num_hidden, root_weight=False
        )
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E

class EdgeMLP(nn.Module):
    """
    define MLP operation for edge updates
    """
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E

class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
            nn.Linear(num_hidden,num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden,num_hidden),
            nn.Sigmoid()
        )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V

class Graph_encoder(nn.Module):
    """
    construct the graph encoder module
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim,
                 seq_in=False, num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim += 20

        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)

        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
                GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers))


    def forward(self, h_V, edge_index, h_E, seq, batch_id):
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = torch.cat([h_V, seq], dim=-1)

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)

        return h_V

class Attention(nn.Module):
    """
    define the attention module
    """
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				# input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  	# x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  					# x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		# attention.shape = (1, attention_hops, seq_len)
        return attention

class DeeppH(nn.Module):
    """
    construct the DeeppH model
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_layers, augment_eps, task, device):
        super(DeeppH, self).__init__()
        self.augment_eps = augment_eps
        self.hidden_dim = hidden_dim
        self.device = device
        self.task = task

        # define the encoder layer
        self.EGNN_encoder = EGNN(
            in_node_nf=node_input_dim,
            in_edge_nf=edge_input_dim,
            out_node_nf=hidden_dim,
            hidden_nf=hidden_dim,
            n_layers=num_layers,
            device=self.device,
            attention=True,
            normalize=True,
            tanh=True
        )
        self.seq_feat_proj = nn.Linear(1024, hidden_dim) # 1024 from prott5
        # define the attention layer
        self.attention = Attention(hidden_dim*2, dense_dim=16,n_heads=4)

        self.add_module("FC_{}1".format(task), nn.Linear(hidden_dim*2, hidden_dim, bias=True))
        self.add_module("FC_{}2".format(task), nn.Linear(hidden_dim, 2, bias=True))

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, structure_feat, seq_feat, edge_index, batch_id):
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X) # [#node, 5 ("N", "Ca", "C", "O", "R"), 3(x,y,z)
            structure_feat = structure_feat + self.augment_eps * torch.randn_like(structure_feat)
        # get the geometric features
        h_V_geo, h_E = get_geo_feat(X, edge_index) # _, [#edge, 450]
        print('h_V_geo, h_E',h_V_geo.shape, h_E.shape)
        structure_feat = torch.cat([structure_feat, h_V_geo], dim=-1) # [#node, 1217]  --> 205
        print('structure_feat',structure_feat.shape)
        structure_feat, x = self.EGNN_encoder(structure_feat, X[:, 1, :], edge_index, h_E)
        print('structure_feat_egnn',structure_feat.shape)
        seq_feat = self.seq_feat_proj(seq_feat)
        print('seq',seq_feat.shape)
        feature_embedding = torch.concat([structure_feat, seq_feat], dim=1)
        batchx = split_batch(feature_embedding, batch_id) # [B,L,hid*2]

        feature_embedding = torch.tensor([]).to(self.device)
        all_attention_weights = []
        for h_vi in batchx:
            # Attention pooling 
            att = self.attention(h_vi) # [1, heads, L]
            all_attention_weights.append(att)
            h_vi = att @ h_vi # [1, heads, hid*2]
            h_vi = torch.sum(h_vi, 1)
            # print("feature_embedding device:", feature_embedding.device)
            # print("h_vi device:", h_vi.device)
            
            feature_embedding = feature_embedding.to(h_vi.device)
            # print("feature_embedding device:", feature_embedding.device)
            # print("h_vi device:", h_vi.device)
            feature_embedding = torch.cat((feature_embedding, h_vi), dim=0)

        emb = F.elu(self._modules["FC_{}1".format(self.task)](feature_embedding))
        output = self._modules["FC_{}2".format(self.task)](emb)

        return output,all_attention_weights

