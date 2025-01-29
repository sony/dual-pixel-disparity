import torch
import torch.nn as nn
from utils.util import MAX_8BIT

loss_names = ['l1', 'l2', 'l1l2', 'l1c', 'l2c', 'l1l2c']

def compute_valid_mask(target):
    return (target > 0).detach()

def compute_diff(pred, target, valid_mask):
    diff = target - pred
    return diff[valid_mask]

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = compute_valid_mask(target)
        diff = compute_diff(pred, target, valid_mask)
        self.loss = (diff**2).mean()
        return self.loss

def L1(pred, target):
    assert pred.dim() == target.dim(), "inconsistent dimensions"
    valid_mask = compute_valid_mask(target)
    diff = compute_diff(pred, target, valid_mask)
    loss = diff.abs().mean()
    return loss    

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        if isinstance(pred, list):
            loss = sum(L1(p, target) for p in pred)
        else:
            loss = L1(pred, target)
        return loss

class MaskedL1L2Loss(nn.Module):
    def __init__(self):
        super(MaskedL1L2Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = compute_valid_mask(target)
        diff = compute_diff(pred, target, valid_mask)
        l1 = diff.abs().mean()
        l2 = (diff**2).mean()
        self.loss = l1 + l2
        return self.loss

class UncertaintyL1Loss(nn.Module):
    def __init__(self):
        super(UncertaintyL1Loss, self).__init__()

    def forward(self, pred, target, conf_inv, conf_lambda):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = compute_valid_mask(target)
        conf_inv = conf_inv * conf_lambda
        diff = torch.sqrt((target - pred)**2 / (conf_inv**2) + 4*torch.log1p(conf_inv))
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class UncertaintyL2Loss(nn.Module):
    def __init__(self):
        super(UncertaintyL2Loss, self).__init__()

    def forward(self, pred, target, conf_inv, conf_lambda):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = compute_valid_mask(target)
        conf_inv = conf_inv * conf_lambda
        diff = (target - pred)**2 / (conf_inv**2) + 4*torch.log1p(conf_inv)
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class UncertaintyL1L2Loss(nn.Module):
    def __init__(self):
        super(UncertaintyL1L2Loss, self).__init__()

    def forward(self, pred, target, conf_inv, conf_lambda):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = compute_valid_mask(target)
        conf_inv = conf_inv * conf_lambda
        l2 = (target - pred)**2 / (conf_inv**2) + 4*torch.log1p(conf_inv)
        l1 = torch.sqrt(l2)
        l2 = l2[valid_mask].abs().mean()
        l1 = l1[valid_mask].abs().mean()
        self.loss = l1 + l2
        return self.loss