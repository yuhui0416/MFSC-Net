import torch

class STLoss(torch.nn.Module):
    def __init__(self):
        super(STLoss, self).__init__()


    def forward(self, pre, label, mask):

        pre = pre*mask
        label = label*mask

        #quebao yanmowai xiangsu wei0
        # pre[~mask] = 0
        # label[~mask] = 0

        error = pre - label
        squared = error**2
        squared_sum = squared.sum()

        n = mask.sum()
        if n != 0:
            loss = squared_sum/n
        else:
            return 0
        return loss