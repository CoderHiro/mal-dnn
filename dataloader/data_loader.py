import json
import os
import numpy as np
import torch
import torch.nn as nn

class file_graph(object):
    def __init__(self, f_num=0, label=None, file_name=None, max_node_num=0, cfg_features=[], cfg_struc=[], total_struc=[]):
        self.file_name = file_name
        self.label = label
        self.max_node_num = max_node_num
        self.cfg_features = cfg_features
        self.cfg_struc = cfg_struc
        self.total_struc = total_struc

def get_features(function_featrues):
    new_features = []
    for block_features in function_featrues:
        block_new_features = block_features[:7]
        new_features.append(block_new_features)
    return new_features

def devide_data(data, shuffle=True):
    train_data = []
    dev_data = []
    test_data = []
    if shuffle == True:
        np.random.shuffle(data)
    for i in range(len(data)):
        if i < len(data)*0.8:
            train_data.append(data[i])
        elif i < len(data)*0.9:
            dev_data.append(data[i])
        else:
            test_data.append(data[i])
    return train_data, dev_data, test_data

def load_data(path, label):

    assert os.path.exists(path)
    
    graphs = []

    for file_name in os.listdir(path):
        file_path = path + file_name
        graph = file_graph(f_num=0, label=None, file_name=None, max_node_num=0, cfg_features=[], cfg_struc=[], total_struc=[])
        graph.label = label
        graph.file_name = file_name
        max_node_num = 0
        with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
            for line in f:
                funcion_info = json.loads(line.strip())
                graph.cfg_features.append(funcion_info['features'])
                # graph.cfg_features.append(get_features(funcion_info['features']))
                graph.cfg_struc.append(funcion_info['succs'])
                graph.total_struc.append(funcion_info['next'])
                if funcion_info['n_num'] > max_node_num:
                    max_node_num = funcion_info['n_num']
        graph.max_node_num = max_node_num
        print("\r", len(graph.total_struc), end="")
        if len(graph.total_struc)*graph.max_node_num*graph.max_node_num < 1024*1024*1024:
            graphs.append(graph)

    return graphs

def get_batch(dataset, batch_size, shuffle=False):
    if shuffle:
        np.random.shuffle(dataset)

    nb_batch = int(np.ceil(len(dataset) / batch_size))  # 向上取整
    for i in range(nb_batch):
        batch_data = dataset[i*batch_size: (i+1)*batch_size]
        if shuffle:
            np.random.shuffle(batch_data)

        yield batch_data

def batch_variable(graphs, args):
    max_block_num = 0
    max_function_num = 0
    batch_size = len(graphs)
    for graph in graphs:
        if graph.max_node_num > max_block_num:
            max_block_num = graph.max_node_num
        if len(graph.total_struc) > max_function_num:
            max_function_num = len(graph.total_struc)

    features = torch.zeros(batch_size, max_function_num, max_block_num, args.block_dim)
    cfg_masks = torch.zeros(batch_size, max_function_num, max_block_num, max_block_num)
    fcg_masks = torch.zeros(batch_size, max_function_num, max_function_num)
    batch_y = torch.zeros(batch_size, dtype=torch.long)

    for i, graph in enumerate(graphs):

        for j, function_featrues in enumerate(graph.cfg_features):
            for k, block_features in enumerate(function_featrues):
                features[i, j, k, :] = torch.tensor(block_features)

        for j, function_struc in enumerate(graph.cfg_struc):
            for k, block_succs in enumerate(function_struc):
                for succs in block_succs:
                    cfg_masks[i, j, k, succs] = 1
        
        for j, function_succs in enumerate(graph.cfg_struc):
            for k, block_succs in enumerate(function_struc):
                cfg_masks[i, j, k, block_succs] = 1
        
        batch_y[i] = graph.label
    
    return features, cfg_masks, fcg_masks, batch_y
