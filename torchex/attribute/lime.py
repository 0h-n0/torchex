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
        explanation = self.explainer.explain_instance(image,
                                                      self.model,
                                                      top_labels=5,
                                                      hide_color=0,
                                                      num_samples=1000)        
