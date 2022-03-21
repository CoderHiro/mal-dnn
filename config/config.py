import os
import argparse
import json

def data_path_parse(path):
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as fin:
        opts = json.load(fin)

    # logging.info(opts)
    print(opts)

    return opts

def arg_parse():
    parser = argparse.ArgumentParser(description="CNN Arguments Configuration")
    # CUDA
    parser.add_argument('--cuda', type=int, default=1, help='-1 means train on CPU')
    # parser.add_argument('--use_cuda', type=bool, default=True, help='use GPU or not')
    parser.add_argument('--enable_cuda', type=bool, default=False, help='enable GPU or not')

    # parameters for cfg_gnn
    parser.add_argument('--block_dim', type=int, default=16, help='The feature dimension of block')
    parser.add_argument('--cfg_embed_dim', type=int, default=32, help='The hidden layer size of cfg-gnn')
    parser.add_argument('--cfg_output_dim', type=int, default=32, help='The output size of cfg-gnn')
    parser.add_argument('--cfg_embed_depth', type=int, default=2, help='The embed depth of cfg-gnn')
    parser.add_argument('--cfg_iter_times', type=int, default=5, help='The iteration times of cfg-gnn')
    # parser.add_argument('--cfg_batch_size', type=int, default=16, help='The batch size of cfg-gnn')

    # parameters for fcg_gnn
    parser.add_argument('--function_dim', type=int, default=32, help='The feature dimension of function')
    parser.add_argument('--fcg_embed_dim', type=int, default=64, help='The hidden layer size of fcg-gnn')
    parser.add_argument('--fcg_output_dim', type=int, default=64, help='The output size of fcg-gnn')
    parser.add_argument('--fcg_embed_depth', type=int, default=2, help='The embed depth of fcg-gnn')
    parser.add_argument('--fcg_iter_times', type=int, default=5, help='The iteration times of fcg-gnn')
    parser.add_argument('--fcg_batch_size', type=int, default=1, help='The batch size of fcg-gnn')

    # parameters for classifier
    parser.add_argument('--hidden', type=int, default=16, help='The hidden layer size of MLP')
    parser.add_argument('--output_dim', type=int, default=2, help='The output_size of MLP')
    
    parser.add_argument('--learning_rate', type=int, default=3e-3, help='The learning rate of model')
    parser.add_argument('--epochs', type=int, default=50, help='The number of epoch')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-7, help='weight decay')

    args = parser.parse_args()

    print(vars(args))

    return args
