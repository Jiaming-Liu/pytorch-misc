import torch
from torch.autograd import Function

# NOT USEFUL! THE OFFICIAL IMPLEMENTATION IS CORRECT.

# Example:
# output = model(data)
# loss = SoftmaxWithCrossEntropy(output, target)
# loss.backward()
# optimizer.step()

def SoftmaxWithCrossEntropy(input, label):
    return _SoftmaxWithCrossEntropy()(input, label)


class _SoftmaxWithCrossEntropy(Function):
    def forward(self, input, label):
        self.save_for_backward(input, label)
        return torch.nn.CrossEntropyLoss()(
            torch.autograd.Variable(input, requires_grad=False),
            torch.autograd.Variable(label, requires_grad=False)
        ).data

    def backward(self, sth):
        input, label = self.saved_tensors
        grad_fs = grad_label = None

        if self.needs_input_grad[0]:
            fs = torch.nn.Softmax()(
                torch.autograd.Variable(input, requires_grad=False)
            ).data

            # neg. one hot label
            y = input.new().resize_as_(input).zero_()
            for i, l in enumerate(label):
                y[i, l] = -1.

            fs.add_(y).mul_(1. / len(label))
            grad_fs = fs
        if self.needs_input_grad[1]:
            raise NotImplementedError()

        return grad_fs, grad_label
