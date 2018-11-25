import torch
import torch.nn as nn
import numpy as np


class IntegratedGradients(object):
    '''

    Reference: https://arxiv.org/abs/1703.01365
    '''
    def __init__(self,
                 model: nn.Module,
                 divided_number: int=100):
        self.model = model
        self.divided_number = divided_number

    def __call__(self, x, label_idx: int, x_base=None):
        self.model.train()
        ig = torch.zeros_like(x)
        
        if x_base is None:
            x_base = torch.zeros_like(x)
            
        if not x.requires_grad:
            x.requires_grad = True
            
        for k in range(self.divided_number):
            self.model.zero_grad()
            x.grad = torch.zeros_like(x)
            alpha = (k + 1) / float(self.divided_number)
            y = self.model(x_base + (x - x_base) * alpha)
            y[:, label_idx].backward()
            ig += (x - x_base) * x.grad / self.divided_number
        self.ig = ig
        return ig

    def check(self):
        return torch.sum(self.ig)
    
    @classmethod
    def heatmap(cls, x:torch.Tensor, filename):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        x = x.data.numpy()
        
        plt.switch_backend('Agg')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        absmax = np.max(np.abs(x))
        im = ax.imshow(x, aspect='equal', cmap=plt.get_cmap('bwr'),
                       vmin=-absmax, vmax=absmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.savefig(filename)
        
        
        
