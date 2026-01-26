import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import MoELayer, smallest_final_layer, one_router_same_expert
from .moe import linearnet1e, nomoe, no_pair_e

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class RT(nn.Module):
    '''
    This is pytorch implementation of RT model.
    We implement the model based on the github repo:
    https://github.com/CameronDiao/relational-transformer
    '''
    def __init__(
            self,
            num_layers: int = 3,
            num_heads: int = 4,
            node_dim: int = 64,
            node_hidden_dim: int = 128,
            edge_dim: int = 64,
            edge_hidden_dim_1: int = 128,
            edge_hidden_dim_2: int = 128,
            dropout: float = 0.0,
            token_wise_routing: bool = False,
    ):
        super(RT, self).__init__()
        if smallest_final_layer:
            self.layers = []
            for i in range(num_layers):
                self.layers.append(RTTransformerLayer(
                    num_heads,
                    node_dim,
                    int(node_hidden_dim / (2**i)),
                    edge_dim,
                    edge_hidden_dim_1,
                    int(edge_hidden_dim_2 / (2**i)),
                    dropout,
                    token_wise_routing=token_wise_routing,
                ))
            self.layers = nn.ModuleList(self.layers)
        else:
            layer = RTTransformerLayer(
                num_heads,
                node_dim,
                node_hidden_dim,
                edge_dim,
                edge_hidden_dim_1,
                edge_hidden_dim_2,
                dropout,
                token_wise_routing=token_wise_routing,
            )
            self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        # self.aggregation = aggregation # 'cat', 'avg', 'sum', 'att'
        self.aggregation = 'cat'
        print('[INFO] PRT Agg: {}'.format(self.aggregation))
        self.node2edge_mlp = MLP(input_dim=node_dim * 2 if self.aggregation == 'cat' else node_dim, output_dim=node_dim, hidden_size=(node_hidden_dim,))
        if self.aggregation == 'att':
            self.attention_mlp = MLP(input_dim=node_dim + edge_dim, output_dim=1, hidden_size=(32,))

    def init_adj(self, num_nodes, batch):
        off_diag = np.ones([num_nodes, num_nodes])
        
        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
        
        rel_rec = torch.FloatTensor(rel_rec).cuda()
        rel_send = torch.FloatTensor(rel_send).cuda()
        
        rel_rec = rel_rec[None, :, :].repeat(batch, 1, 1)
        rel_send = rel_send[None, :, :].repeat(batch, 1, 1)
        
        # rel_rec = rel_rec.reshape(batch, num_nodes, num_nodes, -1)
        # rel_send = rel_send.reshape(batch, num_nodes, num_nodes, -1)
        
        return rel_rec, rel_send

    def init_edge(self, x, rel_rec, rel_send):
        B, N, _ = x.shape
        if self.aggregation == 'cat':
            receivers = torch.matmul(rel_rec, x)
            senders = torch.matmul(rel_send, x)
            edges = self.node2edge_mlp(torch.cat([receivers, senders], dim=-1))
            
        elif self.aggregation == 'avg':
            H = rel_rec + rel_send
            edges = torch.matmul(H, x) / 2
            edges = self.node2edge_mlp(edges)

        elif self.aggregation == 'sum':
            H = rel_rec + rel_send
            edges = torch.matmul(H, x)
            edges = self.node2edge_mlp(edges)
            
        elif self.aggregation == 'att':
            H = rel_rec + rel_send
            x = self.node2edge_mlp(x)
            edge_init = torch.matmul(H, x)
            E = edge_init.shape[1]
            x_rep = (x[:, :, None, :].transpose(2, 1)).repeat(1, E, 1, 1)
            edge_rep = edge_init[:, :, None, :].repeat(1, 1, N, 1)
            node_edge_cat = torch.cat((x_rep, edge_rep), dim=-1)
            attention_weight = self.attention_mlp(node_edge_cat)[:, :, :, 0]
            H_weight = attention_weight * H
            H_weight = F.softmax(H_weight, dim=2)
            H_weight = H_weight * H
            edges = torch.matmul(H_weight, x)
            
        edges = edges.reshape(B, N, N, -1)
            
        return edges
        
    def forward(self, node_features, edge_features_, return_edge=False, epoch=None):
        batch = node_features.shape[0]
        num_nodes = node_features.shape[1]
        
        rel_rec, rel_send = self.init_adj(num_nodes, batch)
        edge_features = self.init_edge(node_features, rel_rec, rel_send)
        
        scores_n = []
        scores_e = []
        logits_n = []
        logits_e = []
        prev_gating_scores = None
        for layer in self.layers:
            node_features, edge_features, score, logit = layer(node_features, edge_features, epoch=epoch, prev_gating_scores=prev_gating_scores)
            if one_router_same_expert:
                prev_gating_scores = score
            scores_n.append(score[0])
            scores_e.append(score[1])
            logits_n.append(logit[0])
            logits_e.append(logit[1])

        if return_edge:
            if scores_n[0] is not None:
                scores = (torch.stack(scores_n), torch.stack(scores_e))
                logits = (torch.stack(logits_n), torch.stack(logits_e))
            else:
                scores = None
                logits = None
            return node_features, edge_features, scores, logits

        return node_features


class RTNoEdgeInit(nn.Module):
    def __init__(
            self,
            num_layers: int = 3,
            num_heads: int = 4,
            node_dim: int = 64,
            node_hidden_dim: int = 128,
            edge_dim: int = 64,
            edge_hidden_dim_1: int = 128,
            edge_hidden_dim_2: int = 128,
            dropout: float = 0.0,
            token_wise_routing: bool = False,
    ):
        super(RTNoEdgeInit, self).__init__()
        layer = RTTransformerLayer(
            num_heads,
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout,
            token_wise_routing=token_wise_routing,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, node_features, edge_features, return_edge=False, epoch=None):
        scores_n = []
        scores_e = []
        logits_n = []
        logits_e = []
        prev_gating_scores = None
        for layer in self.layers:
            node_features, edge_features, score, logit = layer(node_features, edge_features, epoch=epoch, prev_gating_scores=prev_gating_scores)
            if one_router_same_expert:
                prev_gating_scores = score
            scores_n.append(score[0])
            scores_e.append(score[1])
            logits_n.append(logit[0])
            logits_e.append(logit[1])
        
        if return_edge:
            if scores_n[0] is not None:
                scores = (torch.stack(scores_n), torch.stack(scores_e))
                logits = (torch.stack(logits_n), torch.stack(logits_e))
            else:
                scores = None
                logits = None
            return node_features, edge_features, scores, logits

        return node_features


class RTTransformerLayer(nn.Module):
    def __init__(
        self,
        num_heads: int = 4,
        node_dim: int = 64,
        node_hidden_dim: int = 128,
        edge_dim: int = 64,
        edge_hidden_dim_1: int = 128,
        edge_hidden_dim_2: int = 128,
        dropout: float = 0.0,
        token_wise_routing: bool = False,
    ):
        super(RTTransformerLayer, self).__init__()
        self.edge_update = True
        self.attention_layer = RTAttentionLayer(num_heads, node_dim, edge_dim)
        
        self.dropout = nn.Dropout(dropout)
        if nomoe:
            self.linear_net_n = \
                nn.Sequential(
                    nn.Linear(node_dim, node_hidden_dim),
                    # nn.Dropout(dropout),
                    nn.ReLU(inplace=True),
                    nn.Linear(node_hidden_dim, node_dim),
            )
        else:
            self.linear_net_n = \
                MoELayer(node_dim, 
                    node_hidden_dim, 
                    node_dim,
                    token_wise_routing=token_wise_routing)
        
        self.norm1_n = nn.LayerNorm(node_dim)
        self.norm2_n = nn.LayerNorm(node_dim)

        if self.edge_update:
            if linearnet1e:
                self.linear_net1_e = \
                    MoELayer(node_dim * 2 + edge_dim * 2, 
                        edge_hidden_dim_1, 
                        edge_dim,
                        token_wise_routing=token_wise_routing)

                self.linear_net2_e = \
                    nn.Sequential(
                        nn.Linear(edge_dim, edge_hidden_dim_2),
                        # nn.Dropout(dropout),
                        nn.ReLU(inplace=True),
                        nn.Linear(edge_hidden_dim_2, edge_dim),
                    )
            elif nomoe or no_pair_e:
                self.linear_net1_e = \
                    nn.Sequential(
                        nn.Linear(node_dim * 2 + edge_dim * 2, edge_hidden_dim_1),
                        # nn.Dropout(dropout),
                        nn.ReLU(inplace=True),
                        nn.Linear(edge_hidden_dim_1, edge_dim),
                    )   
                self.linear_net2_e = \
                    nn.Sequential(
                        nn.Linear(edge_dim, edge_hidden_dim_2),
                        # nn.Dropout(dropout),
                        nn.ReLU(inplace=True),
                        nn.Linear(edge_hidden_dim_2, edge_dim),
                    )
            else:    
                self.linear_net1_e = \
                    nn.Sequential(
                        nn.Linear(node_dim * 2 + edge_dim * 2, edge_hidden_dim_1),
                        # nn.Dropout(dropout),
                        nn.ReLU(inplace=True),
                        nn.Linear(edge_hidden_dim_1, edge_dim),
                    )

                self.linear_net2_e = \
                    MoELayer(edge_dim, 
                        edge_hidden_dim_2, 
                        edge_dim,
                        token_wise_routing=token_wise_routing)

            
            self.norm1_e = nn.LayerNorm(edge_dim)
            self.norm2_e = nn.LayerNorm(edge_dim)
    
    def forward(self, node_features, edge_features, epoch=None, prev_gating_scores=None):
        _, N, _ = node_features.shape
        attn_out = self.attention_layer(node_features, edge_features)
        node_features = node_features + self.dropout(attn_out)
        node_features = self.norm1_n(node_features)
        
        ########## Node MLP with MoE ##########
        if nomoe:
            linear_out = self.linear_net_n(node_features)
            scores_n, logits_n = None, None
        else:
            linear_out, scores_n, logits_n = self.linear_net_n(node_features, epoch=epoch, prev_gating_scores=prev_gating_scores)
        node_features = node_features + self.dropout(linear_out)
        node_features = self.norm2_n(node_features)
        
        if self.edge_update:
            source_nodes = node_features.unsqueeze(1)
            expanded_source_nodes = source_nodes.repeat(1, N, 1, 1)
            
            target_nodes = node_features.unsqueeze(2)
            expanded_target_nodes = target_nodes.repeat(1, 1, N, 1)
            
            reversed_edge_features = edge_features.permute(0, 2, 1, 3)
            
            concatenated_inputs = torch.cat([edge_features, reversed_edge_features, expanded_source_nodes, expanded_target_nodes], dim=-1)
            # import pdb; pdb.set_trace()
            if linearnet1e:
                ########## Edge MLP with MoE ##########
                edge_features, scores_e, logits_e = self.linear_net1_e(concatenated_inputs, epoch=epoch, prev_gating_scores=prev_gating_scores)
                edge_features = edge_features + self.dropout(edge_features)
                edge_features = self.norm1_e(edge_features)
                
                edge_features = edge_features + self.dropout(self.linear_net2_e(edge_features))
                edge_features = self.norm2_e(edge_features)
            elif nomoe or no_pair_e:
                edge_features = edge_features + self.dropout(self.linear_net1_e(concatenated_inputs))
                edge_features = self.norm1_e(edge_features)
                
                edge_features = edge_features + self.dropout(self.linear_net2_e(edge_features))
                edge_features = self.norm2_e(edge_features)
                (scores_n, scores_e), (logits_n, logits_e) = (None, None), (None, None)
            else:
                edge_features = self.linear_net1_e(concatenated_inputs)
                edge_features = edge_features + self.dropout(edge_features)
                edge_features = self.norm1_e(edge_features)
                
                ########## Edge MLP with MoE ##########
                edge_features, scores_e, logits_e = self.linear_net2_e(edge_features, epoch=epoch, prev_gating_scores=prev_gating_scores)
                edge_features = edge_features + self.dropout(edge_features)
                edge_features = self.norm2_e(edge_features)

        # import pdb; pdb.set_trace()
        
        return node_features, edge_features, (scores_n, scores_e), (logits_n, logits_e)


class RTAttentionLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        node_dim: int,
        edge_dim: int,
    ):
        super(RTAttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        
        self.proj_qkv_n = nn.Linear(node_dim, 3 * node_dim)
        self.proj_qkv_e = nn.Linear(edge_dim, 3 * node_dim)
        self.proj_o = nn.Linear(node_dim, node_dim)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, node_features, edge_features):
        B, N, _ = node_features.shape
        qkv_n = self.proj_qkv_n(node_features)
        qkv_e = self.proj_qkv_e(edge_features)
        
        qkv_n = qkv_n.reshape(B, N, self.num_heads, 3 * self.head_dim)
        qkv_n = qkv_n.permute(0, 2, 1, 3)
        q_n, k_n, v_n = qkv_n.chunk(3, dim=-1)
        
        qkv_e = qkv_e.reshape(B, N, N, self.num_heads, 3 * self.head_dim)
        qkv_e = qkv_e.permute(0, 3, 1, 2, 4)
        q_e, k_e, v_e = qkv_e.chunk(3, dim=-1)
        
        q = q_n.reshape(B, self.num_heads, N, 1, self.head_dim) + q_e
        k = k_n.reshape(B, self.num_heads, 1, N, self.head_dim) + k_e
        
        q = q.reshape(B, self.num_heads, N, N, 1, self.head_dim)
        k = q.reshape(B, self.num_heads, N, N, self.head_dim, 1)
        
        qk = torch.matmul(q, k)
        qk = qk.reshape(B, self.num_heads, N, N)
        
        qk = qk * self.scale
        att_dist = F.softmax(qk, dim=-1)
        
        att_dist = att_dist.reshape(B, self.num_heads, N, 1, N)
        v = v_n.reshape(B, self.num_heads, 1, N, self.head_dim) + v_e
        
        new_nodes = torch.matmul(att_dist, v)
        new_nodes = new_nodes.reshape(B, self.num_heads, N, self.head_dim)
        new_nodes = new_nodes.permute(0, 2, 1, 3)
        new_nodes = new_nodes.reshape(B, N, self.num_heads * self.head_dim)
        new_nodes = self.proj_o(new_nodes)
        
        return new_nodes
