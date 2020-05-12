import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

class Timer:
    #
    def __init__(self):
        self.total = time.time()
        self.start = time.time()

    def get(self):
        tmp = self.start
        self.start = time.time()
        return self._get(tmp)

    def getTotal(self):
        tmp = self.total
        self.total = time.time()
        return self._get(tmp)

    def _get(self, begin):
        end = time.time()
        hours, rem = divmod(end - begin, 3600)
        minutes, seconds = divmod(rem, 60)
        return '{h:0>2}:{m:0>2}:{s:0>2}'.format(h=int(hours), m=int(minutes), s=int(seconds))


class DataContainer(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.Y[index]
        return X, y
		
class Common():

    @staticmethod
    def optimizer(model, lr):
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    @staticmethod
    def criterion():
        # F.log_softmax + nn.NLLLoss
        return nn.CrossEntropyLoss()

    @staticmethod
    def send_device(device, items):
        if isinstance(items, torch.Tensor):
            return items.to(device)
        if isinstance(items, list):
            out = torch.stack(items, dim=0).to(device)
            return out

    @staticmethod
    def accuracy(scores, y, device):
        return np.mean(np.argmax(scores.detach().cpu().numpy(), axis=1) == y.detach().cpu().numpy())
        #sum(torch.argmax(scores, axis=1) == y.to(device)).true_divide(len(y)).item()

    @staticmethod
    def generator(X, Y, B):
        return DataLoader(DataContainer(X, Y), batch_size=B, shuffle=True)

    @staticmethod
    def save_checkpoint(states, model_file):
        torch.save(states, model_file)

    @staticmethod
    def load_checkpoint(model_file):
        return torch.load(model_file)

    @staticmethod
    def device():
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
