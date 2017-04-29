from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm

# Example:
#    model = My100000LayerCNNPublishedIn2030(num_classes=10000000)
#    from weight_grouper import group_weight
#    params=group_weight(model)
#    optimizer = torch.optim.SGD(params, 0.1,
#                                momentum=0.9,
#                                weight_decay=0.0005,nesterov=True)


def group_weight(module):
    group_decay = []
    group_no_decay = []

    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, _ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, _BatchNorm):
            if m.bias is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups
