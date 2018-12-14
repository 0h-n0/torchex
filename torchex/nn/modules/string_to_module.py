import torch.nn 

def get_module(module_name: str):
    """only support pytorch.nn modules
    """
    module_name = module_name.lower()
    print(torch.nn.modules.__all__)
    


if __name__ == '__main__':
    get_module('linear')
