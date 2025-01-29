import torch.nn as nn

class GT(nn.Module):
    def __init__(self, args):
        super(GT, self).__init__()
        self.args = args

    def forward(self, input):
        gt = input['gt']
        return gt
    
class Through(nn.Module):
    def __init__(self, args):
        super(Through, self).__init__()
        self.args = args

    def forward(self, input):
        d = input['d']
        return d