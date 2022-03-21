import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import Cfg_Gnn, Fcg_Gnn

class Classifier(nn.Module):
    def __init__(self, args):

        super(Classifier, self).__init__()
        self.cfg_gnn = Cfg_Gnn( 
            N_x = args.block_dim, 
            N_embed = args.cfg_embed_dim, 
            N_out = args.cfg_output_dim, 
            depth_embed = args.cfg_embed_depth, 
            iter_level = args.cfg_iter_times,
            devices = args.device
        )
        self.fcg_gnn = Fcg_Gnn(
            N_x = args.function_dim, 
            N_embed = args.fcg_embed_dim, 
            N_out = args.fcg_output_dim, 
            depth_embed = args.fcg_embed_depth, 
            iter_level = args.fcg_iter_times,
            devices = args.device
        )

        self.hidden = torch.nn.Linear(args.fcg_output_dim, args.hidden)   # hidden layer
        self.out = torch.nn.Linear(args.hidden, args.output_dim)   # output layer

    def forward(self, X, cfg_masks, fcg_mask):

        function_feature = self.cfg_gnn(X, cfg_masks)

        file_feature = self.fcg_gnn(function_feature, fcg_mask)

        hidden_1 = self.hidden(file_feature)

        output = self.out(hidden_1)

        return output