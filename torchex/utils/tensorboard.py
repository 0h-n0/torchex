from tensorboardX import SummaryWriter


class TensorBoarder(object):
    def __init__(self):
        self.writer = SummaryWriter()

    def histgram(self, module, n_iter):
        for name, param in module.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)


