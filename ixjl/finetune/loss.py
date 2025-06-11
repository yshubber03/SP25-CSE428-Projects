import torch.nn as nn
#Please check https://pytorch.org/docs/stable/nn.html for more loss functions
import torch.nn.functional as F
from pytorch_msssim import ssim # Make sure you install this: pip install pytorch-msssim

class cosine_distance(nn.Module):
    def __init__(self):
        super(cosine_distance, self).__init__()
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-08)
    def forward(self, x, y):
        return 1-self.cos(x,y)

def configure_loss(args):
    if args.loss_type == 1:
        return nn.MSELoss()
    elif args.loss_type == 2:
        return cosine_distance()
    elif args.loss_type == 3:
        return CombinedMSESSIMLoss(alpha=0.8)
    else:
        raise Exception("Unknown loss type: {}".format(args.loss_type)) 

class CombinedMSESSIMLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(CombinedMSESSIMLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss



