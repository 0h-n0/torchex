import torch.nn as nn

class LSTMStep(nn.Module):
    """Layer that performs a single step LSTM update.
    This layer performs a single step LSTM update. Note that it is *not*
    a full LSTM recurrent network. The LSTMStep layer is useful as a
    primitive for designing layers such as the AttnLSTMEmbedding or the
    IterRefLSTMEmbedding below.
    """
    def __init__(self,
                 output_dim,
                 input_dim,
                 init_fn=initializations.glorot_uniform,
                 inner_init_fn=initializations.orthogonal,
                 activation_fn=activations.tanh,
                 inner_activation_fn=activations.hard_sigmoid,
                 **kwargs):
        pass


class AttnLSTMEmbedding(nn.Module):
    """Implements AttnLSTM as in matching networks paper.
    The AttnLSTM embedding adjusts two sets of vectors, the "test" and
    "support" sets. The "support" consists of a set of evidence vectors.
    Think of these as the small training set for low-data machine
    learning.  The "test" consists of the queries we wish to answer with
    the small amounts ofavailable data. The AttnLSTMEmbdding allows us to
    modify the embedding of the "test" set depending on the contents of
    the "support".  The AttnLSTMEmbedding is thus a type of learnable
    metric that allows a network to modify its internal notion of
    distance.
    References:
    Matching Networks for One Shot Learning
    https://arxiv.org/pdf/1606.04080v1.pdf
    Order Matters: Sequence to sequence for sets
    https://arxiv.org/abs/1511.06391
    """

    def __init__(self, n_test, n_support, n_feat, max_depth, **kwargs):
        pass


class IterRefLSTMEmbedding(nn.Module):
    """Implements the Iterative Refinement LSTM.
    Much like AttnLSTMEmbedding, the IterRefLSTMEmbedding is another type
    of learnable metric which adjusts "test" and "support." Recall that
    "support" is the small amount of data available in a low data machine
    learning problem, and that "test" is the query. The AttnLSTMEmbedding
    only modifies the "test" based on the contents of the support.
    However, the IterRefLSTM modifies both the "support" and "test" based
    on each other. This allows the learnable metric to be more malleable
    than that from AttnLSTMEmbeding.
    """
    
    def __init__(self, n_test, n_support, n_feat, max_depth, **kwargs):
        pass
