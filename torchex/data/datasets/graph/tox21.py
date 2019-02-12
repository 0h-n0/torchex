from pathlib import Path

from ..utils import download

class Tox21Dataset(object):
    def __init__(self, root_path='~/.torchex_data/tox21', datatype='train'):
        self.root_path = Path(root_path).expanduser().resolve()
        self.datatype = datatype
        
        assert datatype in ['train', 'val', 'test'], \
            'datatype must be [train, val, test]. ({})'.format(datatype)
        download('https://tripod.nih.gov/tox21/challenge/download?', self.root_path)
        

if __name__ == '__main__':
    d = Tox21Dataset()
