import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import os

from config import config
from dataloader.data_loader import load_data, batch_variable, devide_data
from module.classifier import Classifier
from utils.trainer import Trainer

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
# log_path = os.path.dirname(os.getcwd()) + '/Logs/'
# log_name = log_path + rq + '.log'
# logfile = log_name
# fh = logging.FileHandler(logfile, mode='w')
# fh.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# fh.setFormatter(formatter)
# logger.addHandler(fh)

if __name__ == "__main__":

    np.random.seed(666)
    torch.manual_seed(6666)
    torch.cuda.manual_seed(1234)
    # torch.cuda.manual_seed_all(4321)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    print('GPU available: ', torch.cuda.is_available())
    print('CuDNN available: ', torch.backends.cudnn.enabled)
    print('GPU number: ', torch.cuda.device_count())

    # path import
    path_opts = config.data_path_parse('./config/path.json')

    # import parameters
    args = config.arg_parse()
    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    print(args)

    # load data
    malware_graphs = load_data(path_opts['data']['malware_data'], 1)
    normal_graphs = load_data(path_opts['data']['normal_data'], 0)
    train_graphs, dev_graphs, test_graphs = devide_data(malware_graphs + normal_graphs)
    print("train graphs:{}, dev graphs:{}, test_graphs:{}".format(len(train_graphs), len(dev_graphs), len(test_graphs)))
    # train_graphs = load_data(path_opts['data']['test_data'], 0)
    # dev_graphs = load_data(path_opts['data']['test_data'], 0)

    # inititalize the model
    # model = Classifier(args).to(args.device)

    model = Classifier(args)
    # model = nn.DataParallel(model).to(args.device)
    
    # train
    trainer = Trainer(args, model)
    trainer.train(train_graphs, dev_graphs)
    trainer.evaluate(test_graphs)

    # save
    trainer.save(path_opts['model']['save_model_path'], True)