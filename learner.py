import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

class QErrorLoss:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def __call__(self, input_card, true_card):
        input_card = torch.max(torch.tensor(1.0), input_card)
        part1 = input_card / true_card
        part2 = true_card / input_card
        q_error = torch.max(part1, part2)

        return q_error
    
class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, predictions, targets):

        log_predictions = torch.log(predictions + 1)
        log_targets = torch.log(targets + 1)
        loss = torch.mean((log_predictions - log_targets) ** 2)
        return loss

class Q_GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Q_GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        
        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
    
        return x

class G_GCNLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(G_GCNLayer, self).__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x, adj): 

        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
        
        return x
    
class CrossGraphFormer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(CrossGraphFormer, self).__init__()

        self.d_k = out_dim // num_heads
        self.num_heads = num_heads        

        self.q_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.k_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.v_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))

        self.dropout = nn.Dropout(0.1)

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.q_weight, a = math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_weight, a = math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_weight, a = math.sqrt(5))

    def forward(self, x, q_x):
        num_query = q_x.size(0)
        num_data = x.size(0)

        q = torch.matmul(q_x, self.q_weight)
        k = torch.matmul(x, self.k_weight)
        v = torch.matmul(q_x, self.v_weight) # (num_query, out_dim)

        q = q.view(num_query, self.num_heads, self.d_k).transpose(0, 1)  # (num_heads, num_query, d_k)
        k = k.view(num_data, self.num_heads, self.d_k).transpose(0, 1) # (num_heads, num_data, d_k)
        v = v.view(num_query, self.num_heads, self.d_k).transpose(0, 1) # (num_heads, num_query, d_k)       

        attn_scores = torch.matmul(k, q.transpose(-2, -1)) / math.sqrt(self.d_k)  # (num_heads, num_data, num_query)
        attn_weights = F.softmax(attn_scores, dim=-1) # (num_heads, num_data, num_query)

        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v) # (num_heads, num_data, d_k)
        out = out.transpose(0, 1).contiguous().view(num_data, self.num_heads * self.d_k)  # (num_data, out_dim)
        out = out + x 

        return out
  
class Multi_GINs(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Multi_GINs, self).__init__()

        self.trans_layer = nn.Sequential(nn.Linear(1, out_dim)) 
        self.g_encoder = G_GCNLayer(out_dim, out_dim)
        self.q_encoder = Q_GCNLayer(out_dim, out_dim)
        self.cross_attention = CrossGraphFormer(out_dim, out_dim, 4)
        self.arrg_layer = nn.Sequential(nn.Linear(out_dim, 2*out_dim), nn.LeakyReLU(), nn.Linear(2*out_dim, out_dim)) 
            
    def forward(self, query_generator, data_generator, device):   

        _, x = next(data_generator)
        _, q_x = next(query_generator)

        x = x.to(device)
        q_x = q_x.to(device)

        x = self.trans_layer(x.view(-1,1)) # leaf layer
        q_x = self.trans_layer(q_x.view(-1,1))

        try:
            while True:
                adj, x_in = next(data_generator)
                q_adj, q_x_in = next(query_generator)

                x_in = x_in.to(device)
                adj = adj.to(device)
                q_x_in = q_x_in.to(device)
                q_adj = q_adj.to(device)

                x_in = self.trans_layer(x_in.view(-1,1))
                q_x_in = self.trans_layer(q_x_in.view(-1,1))

                x = torch.cat((x, x_in), dim = 0)
                q_x = torch.cat((q_x, q_x_in), dim = 0)

                x = self.g_encoder(x, adj)
                q_x = self.q_encoder(q_x, q_adj)

                x = self.arrg_layer(x)
                q_x = self.arrg_layer(q_x)                  

                x = self.cross_attention(x, q_x)

        except StopIteration:
            torch.cuda.empty_cache()

        return x, q_x
    
class Flow_Learner(nn.Module):
    def __init__(self, out_dim):
        super(Flow_Learner, self).__init__()
        self.encoder = Multi_GINs(1, out_dim)   
        self.prediction = Regression(out_dim)
    
    def forward(self, query_generator, data_generator, device):

        g_x, q_x= self.encoder(query_generator, data_generator, device)
        pred = self.prediction(q_x, g_x)
            
        return pred

class Regression(nn.Module):
    def __init__(self, out_dim):
        super(Regression, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(2*out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, 1),
            nn.ReLU()
        ) 

    def forward(self, q_feature, g_feature):
        q_feature = q_feature.sum(dim=0)
        g_feature = g_feature.sum(dim=0)

        output = self.linear_layers(torch.cat((q_feature, g_feature), dim=0))  
        pred = output.squeeze()

        return pred 



        