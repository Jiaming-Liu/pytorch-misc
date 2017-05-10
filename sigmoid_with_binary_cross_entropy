import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss, _assert_no_grad


# This class replaces the official (unstable) MultiLabelSoftMarginLoss.

class SigmoidWithBinaryCrossEntropy(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, inplace=False):
        if inplace: # run sigmoid in place
            self.function = _SigmoidWithBinaryCrossEntropyInplace
        else:
            self.function = _SigmoidWithBinaryCrossEntropy
        super(SigmoidWithBinaryCrossEntropy, self).__init__(weight, size_average)

    def forward(self, input, target):
        _assert_no_grad(target)
        return self.function(
            self.weight, self.size_average)(input, target)


class _SigmoidWithBinaryCrossEntropyInplace(Function):
    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        self.size_average = size_average
        super(_SigmoidWithBinaryCrossEntropyInplace, self).__init__()

    def forward(self, input, target):
        input.sigmoid_()
        self.save_for_backward(input, target)
        return F.binary_cross_entropy(
            torch.autograd.Variable(input, requires_grad=False),
            torch.autograd.Variable(target, requires_grad=False),
            weight=self.weight, size_average=self.size_average
        ).data

    def backward(self, sth):
        input, target = self.saved_tensors
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            mult = 1. / (target.size(0) if self.size_average else 1) / (target.size(1) if self.weight is None else 1)
            grad_input = input.sub(target).mul_(mult)
            if self.weight is not None:
                grad_input.mul_(self.weight.view(1, target.size(1)).expand_as(target))
        return grad_input, grad_target


class _SigmoidWithBinaryCrossEntropy(Function):
    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        self.size_average = size_average
        super(_SigmoidWithBinaryCrossEntropy, self).__init__()

    def forward(self, input, target):
        self.save_for_backward(input, target)
        return torch.nn.MultiLabelSoftMarginLoss(weight=self.weight,
                                                 size_average=self.size_average)(
            torch.autograd.Variable(input, requires_grad=False),
            torch.autograd.Variable(target, requires_grad=False),
        ).data

    def backward(self, sth):
        input, target = self.saved_tensors
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            mult = 1. / (target.size(0) if self.size_average else 1) / (target.size(1) if self.weight is None else 1)
            f = input.sigmoid()
            grad_input = f.sub_(target).mul_(mult)
            if self.weight is not None:
                grad_input.mul_(self.weight.view(1, target.size(1)).expand_as(target))
        return grad_input, grad_target
