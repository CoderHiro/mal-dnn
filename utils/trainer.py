import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import torch.optim as optim
import os

from dataloader.data_loader import get_batch, batch_variable

class Trainer(object):
    
    def __init__(self, args, model):
        super(Trainer, self).__init__()
        self.classify_model = model
        self.args = args

    def train(self, train_data, dev_data):
        # optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.classify_model.parameters()),
                               lr=self.args.learning_rate,
                               weight_decay=self.args.weight_decay)

        for i in range(self.args.epochs):
            self.classify_model.train()

            start = time.time()
            train_acc, train_loss = 0, 0
            for j, train_batch_data in enumerate(get_batch(train_data, self.args.fcg_batch_size)):
                # 数据变量(Tensor)化
                features, cfg_masks, fcg_masks, batch_y = batch_variable(train_batch_data, self.args)

                features = features.to(self.args.device)
                cfg_masks = cfg_masks.to(self.args.device)
                fcg_masks = fcg_masks.to(self.args.device)
                batch_y = batch_y.to(self.args.device)

                # 梯度清零
                self.classify_model.zero_grad()

                # 将数据喂给模型，预测输出(前向传播)
                pred = self.classify_model(features, cfg_masks, fcg_masks)

                # 计算误差
                loss = self._calc_loss(pred, batch_y)  # 单元素的tensor
                acc_val = self._calc_acc(pred, batch_y)
                loss_val = loss.data.cpu().item()
                train_loss += loss_val
                train_acc += acc_val
                # print("\r", "[Train] epoch %d, batch %d  loss: %.3f acc: %.3f" % (i+1, j+1, loss_val, acc_val / len(train_batch_data)), end="")

                loss.backward()

                # 更新网络参数
                # for p in classifier.parameters():
                #     p.data -= self._args.learning_rate * p.grad.data

                optimizer.step()

            train_acc /= len(train_data)
            train_loss /= len(train_data)

            dev_acc, dev_loss = self._validate(dev_data)

            end = time.time()

            print("[Epoch%d] time: %.2fs train_loss: %.3f train_acc: %.3f  dev_loss: %.3f dev_acc: %.3f" %
                  (i + 1, (end - start), train_loss, train_acc, dev_loss, dev_acc))


    def _validate(self, dev_data):
        self.classify_model.eval()

        dev_acc, dev_loss = 0, 0
        for k, dev_batch_data in enumerate(get_batch(dev_data, self.args.fcg_batch_size)):
            with torch.no_grad():  # 确保在代码执行期间没有计算和存储梯度, 起到预测加速作用
                features, cfg_masks, fcg_masks, batch_y = batch_variable(dev_batch_data, self.args)

                features = features.to(self.args.device)
                cfg_masks = cfg_masks.to(self.args.device)
                fcg_masks = fcg_masks.to(self.args.device)
                batch_y = batch_y.to(self.args.device)

                pred = self.classify_model(features, cfg_masks, fcg_masks)

                loss = self._calc_loss(pred, batch_y)
                acc_val = self._calc_acc(pred, batch_y)
                loss_val = loss.data.cpu().item()
                dev_acc += acc_val
                dev_loss += loss_val

        dev_acc /= len(dev_data)
        dev_loss /= len(dev_data)
        return dev_acc, dev_loss

    def evaluate(self, test_data):
        self.classify_model.eval()
        test_acc = 0
        for test_batch_data in get_batch(test_data, self.args.fcg_batch_size, shuffle=False):
            features, cfg_masks, fcg_masks, batch_y = batch_variable(test_batch_data, self.args)

            features = features.to(self.args.device)
            cfg_masks = cfg_masks.to(self.args.device)
            fcg_masks = fcg_masks.to(self.args.device)
            batch_y = batch_y.to(self.args.device)

            pred = self.classify_model(features, cfg_masks, fcg_masks)

            acc_val = self._calc_acc(pred, batch_y)
            test_acc += acc_val

        test_acc /= len(test_data)
        print("=== test acc: %.3f ===" % test_acc)
        return test_acc
    
    def _calc_acc(self, pred, target):
        return torch.eq(torch.argmax(pred, dim=1), target).cpu().sum().item()

    def _calc_loss(self, pred, target):
        loss_func = nn.CrossEntropyLoss()  # LogSoftmax + NLLLoss
        loss = loss_func(pred, target)
        return loss

    def save(self, save_path, save_all=False):
        if save_all:
            torch.save(self.classify_model, save_path)
        else:
            torch.save(self.classify_model.state_dict(), save_path)

    def load(self, load_path, load_all=False):
        assert os.path.exists(load_path)
        if load_all:
            self.classify_model = torch.load(load_path, map_location='cpu')
        else:
            self.classify_model.load_state_dict(torch.load(load_path, map_location='cpu'))
        self.classify_model.eval()