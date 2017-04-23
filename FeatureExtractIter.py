import torch
from torch.autograd import Variable
from thread import start_new_thread
try:
    import queue
except ImportError:
    import Queue as queue

    
# Demo:
# 
# from torchvision.models.vgg import make_layers
# from torch.utils.data import DataLoader
# import torchvision.datasets as datasets
# from FeatureExtractIter import FeatureExtractIter
# 
# feature_net = make_layers(
#     [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], False)
# 
# feature_net.cuda(0)
# feature_net.eval()
#
# train_loader = DataLoader(
#     dataset=datasets.ImageFolder('../MY_FAV_DATASET/Train/'),
#     batch_size=256,
#     collate_fn=modified_collate_fn, # see sample of collate_fn in the end of this file
#     shuffle=True,
#     num_workers=8)
# 
# for epoch in range(2):  # loop over the dataset multiple times
# 
#     iter = FeatureExtractIter(train_loader, feature_net, dev_id=0)
#     for i, data in enumerate(iter, 0):
#         pass
#


class FeatureExtractIter(object):
    def __init__(self, dataloader, extractor, dev_id, max_queue_size=2):
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert isinstance(extractor, torch.nn.Module)
        self.q = queue.Queue(maxsize=max_queue_size)
        self.dataloader = dataloader
        start_new_thread(extract_feature, (dataloader, self.q, extractor, dev_id))

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get()
        if obj is None:
            raise StopIteration
        else:
            return obj

    def __len__(self):
        return len(self.dataloader)

    next = __next__


def extract_feature(dataloader, q, extractor, dev_id):
    """ Note that we should change collate_fn so that 
        input2s: [mini-mini-batch1,...,mini-mini-batch4] """
    try:
        for i, (input1, input2s, target) in enumerate(dataloader, 0):
            out = []
            for x in input2s:
                x = Variable(x, volatile=True)
                x = x.cuda(dev_id)
                out.append(extractor(x).cpu().data)
            out = torch.cat(out, 0)
            q.put((input1, out, target))
    except KeyboardInterrupt:
        dataloader._shutdown_workers()
    finally:
        q.put(None)


"""Sample of collate_fn"""
# from torch.utils.data.dataloader import default_collate
# mini_mini_batch_size=64
# def collate_fn(batch):
#     imgs, targets=default_collate(batch)
#     out=[ imgs[i:(i+mini_mini_batch_size)]
#           for i in xrange(0, len(imgs) , mini_mini_batch_size) ]
#     return imgs, out, targets
