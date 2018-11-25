import torch.nn as nn
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm


class Lime(object):
    '''

    '''
    def __init__(self,
                 model: nn.Module):
        self.model = model
        self.explainer = lime_image.LimeImageExplainer(verbose = False)
        self.segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

    def __call__(self, x):
        explanation = self.explainer.explain_instance(x.to('cpu').numpy(),
                                                      self.model,
                                                      top_labels=5,
                                                      hide_color=0,
                                                      num_samples=1)        

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.net = torch.nn.Linear(32*32*3, 10)
        
    def forward(self, x):
        if isinstance(type(x), type(np.ndarray)):
            x = torch.Tensor(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        return self.net(x)
        
if __name__ == '__main__':
    import torch
    import numpy as np
    x = torch.rand(1, 32, 32, 3)
    net = NN()
    net(x)
    lime = Lime(net)
    lime(x[0][0])
