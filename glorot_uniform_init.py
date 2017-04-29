import torch.nn as nn
import math


class MyNet(nn.Module):
    # USE THIS FOR YOUR VISION NETWORK!
    def _initialize_weights(self):
        print('initializing')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                receptive_field_size = m.kernel_size[0] * m.kernel_size[1]
                fansum = (m.out_channels + m.in_channels) * receptive_field_size
                scale = 1. / max(1., float(fansum) / 2.)
                stdv = math.sqrt(3. * scale)
                m.weight.data.uniform_(-stdv, stdv)

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fansum = m.weight.size(1) + m.weight.size(0)
                scale = 1. / max(1., float(fansum) / 2.)
                stdv = math.sqrt(3. * scale)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def __init__(self, num_classes):
        super(MyNet, self).__init__()
        self.features = nn.Sequential(
            ?????????????????????????????????
        )
        self.classifier = nn.Sequential(
            ?????????????????????????????????
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
