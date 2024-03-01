import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP

from torch.utils.checkpoint import checkpoint


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        print(self.in_features)
        print(self.out_features)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h_prime = torch.mm(input, self.W)
        print(input.shape)
        N = h_prime.size()[0]
        
        # if N == 1296:
        #     print("stop")


        a_input = self._prepare_attentional_mechanism_input(h_prime)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        attention = F.softmax(e, dim=1)
        
        torch.cuda.empty_cache()
        # h_prime_attended = torch.bmm(attention.unsqueeze(1), h_prime.unsqueeze(2)).squeeze(1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
        
        return attention

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes
        print(N, Wh)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)



class GraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 # final_dropout,
                 learn_eps,
                 neighbor_pooling_type,
                 device):
        '''
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        device: which device to use
        '''

        super(GraphCNN, self).__init__()

        # self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        # common out the eps if you do not need to use it, otherwise the it will cause
        # error "not in the computational graph"
        if self.learn_eps:
            self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        

        # List of MLPs
        self.mlps = torch.nn.ModuleList()
        
        # List of attation
        self.attention = torch.nn.ModuleList()

        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        
        


        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                self.attention.append(GraphAttentionLayer(in_features = input_dim,
                                             out_features = 36,
                                             alpha = 0.01,
                                             concat=True))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                self.attention.append(GraphAttentionLayer(in_features = hidden_dim,
                                             out_features = 36,
                                             alpha = 0.01,
                                             concat=True))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
    

    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        attention = self.attention[layer](input = h, adj = Adj_block)
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.mm(Adj_block.mul(attention), h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        attention = self.attention[layer](input = h, adj = Adj_block)

        # pooling neighboring nodes and center nodes altogether
        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            # print(Adj_block.dtype)
            # print(h.dtype)
            pooled = torch.mm(Adj_block.mul(attention), h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj):

        x_concat = x
        graph_pool = graph_pool

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = padded_nei
        else:
            Adj_block = adj
            

        # list of hidden representation at each layer (including input)
        h = x_concat

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)

        h_nodes = h.clone()
        # print(graph_pool.shape, h.shape)
        pooled_h = torch.sparse.mm(graph_pool, h)
        # pooled_h = graph_pool.spmm(h)

        return pooled_h, h_nodes

