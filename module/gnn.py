import torch
import torch.nn as nn
import torch.nn.functional as F

class Cfg_Gnn(nn.Module):
    def __init__(self, N_x, N_embed, N_out, depth_embed, iter_level, devices):

        super(Cfg_Gnn, self).__init__()
        self.N_x = N_x
        self.N_embed = N_embed
        self.N_out = N_out
        self.depth_embed = depth_embed
        self.iter_level = iter_level
        self.devices = devices

        self.Wnode = torch.randn((N_x, N_embed), requires_grad=True).to(self.devices)

        self.Wembed = []
        for i in range(depth_embed):
            embed_matrix = torch.randn((N_embed, N_embed), requires_grad=True).to(self.devices)
            self.Wembed.append(embed_matrix)

        # self.W_output = torch.randn((N_embed, N_out), requires_grad=True).to(self.devices)
        # self.b_output = torch.randn((N_out), requires_grad=True).to(self.devices)
        
    def forward(self, X, msg_mask):

        # ([batch_size, node_num, feature_size] -> [batch_size*node_num, feature_size]) * [feature_size, embed_size]
        X_size = tuple(X.size())
        node_val = torch.mm(torch.reshape(X, (-1, self.N_x)), self.Wnode)
        new_size = X_size[:-1] + (self.N_embed,)
        # [batch_size*node_num, embed_size] -> [batch_size, node_num, embed_size]
        node_val = torch.reshape(node_val, new_size)

        cur_msg = F.relu(node_val)
        for t in range(self.iter_level):
            #Message convey
            Li_t = torch.matmul(msg_mask, cur_msg)  #[batch, node_num, embed_dim]
            #Complex Function
            cur_info = torch.reshape(Li_t, (-1, self.N_embed))
            for Wi in self.Wembed:
                if torch.equal(Wi, self.Wembed[-1]):
                    cur_info = torch.mm(cur_info, Wi)
                else:
                    cur_info = F.relu(torch.mm(cur_info, Wi))
            neigh_val_t = torch.reshape(cur_info, Li_t.size())
            #Adding
            tot_val_t = node_val + neigh_val_t
            #Nonlinearity
            tot_msg_t = torch.tanh(tot_val_t)
            cur_msg = tot_msg_t   #[batch, node_num, embed_dim]

        output = torch.sum(cur_msg, -2)   #[batch, embed_dim]
        # output = torch.matmul(g_embed, self.W_output) + self.b_output # [batch, output_dim]

        return output

class Fcg_Gnn(nn.Module):
    def __init__(self, N_x, N_embed, N_out, depth_embed, iter_level, devices):

        super(Fcg_Gnn, self).__init__()
        self.N_x = N_x
        self.N_embed = N_embed
        self.N_out = N_out
        self.depth_embed = depth_embed
        self.iter_level = iter_level
        self.devices = devices

        self.Wnode = torch.randn((N_x, N_embed), requires_grad=True).to(self.devices)

        self.Wembed = []
        for i in range(depth_embed):
            embed_matrix = torch.randn((N_embed, N_embed), requires_grad=True).to(self.devices)
            self.Wembed.append(embed_matrix)

        self.W_output = torch.randn((N_embed, N_out), requires_grad=True).to(self.devices)
        self.b_output = torch.randn((N_out), requires_grad=True).to(self.devices)
        
    def forward(self, X, msg_mask):

        # ([batch_size, node_num, feature_size] -> [batch_size*node_num, feature_size]) * [feature_size, embed_size]
        X_size = tuple(X.size())
        node_val = torch.mm(torch.reshape(X, (-1, self.N_x)), self.Wnode)
        new_size = X_size[:-1] + (self.N_embed,)
        # [batch_size*node_num, embed_size] -> [batch_size, node_num, embed_size]
        node_val = torch.reshape(node_val, new_size)

        cur_msg = F.relu(node_val)
        for t in range(self.iter_level):
            #Message convey
            Li_t = torch.matmul(msg_mask, cur_msg)  #[batch, node_num, embed_dim]
            #Complex Function
            cur_info = torch.reshape(Li_t, (-1, self.N_embed))
            for Wi in self.Wembed:
                if torch.equal(Wi, self.Wembed[-1]):
                    cur_info = torch.mm(cur_info, Wi)
                else:
                    cur_info = F.relu(torch.mm(cur_info, Wi))
            neigh_val_t = torch.reshape(cur_info, Li_t.size())
            #Adding
            tot_val_t = node_val + neigh_val_t
            #Nonlinearity
            tot_msg_t = torch.tanh(tot_val_t)
            cur_msg = tot_msg_t   #[batch, node_num, embed_dim]

        g_embed = torch.sum(cur_msg, -2)   #[batch, embed_dim]
        output = torch.matmul(g_embed, self.W_output) + self.b_output # [batch, output_dim]

        return output