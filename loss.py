import torch
from torch import nn
from torch.autograd import Function

# ================================ Dice BCE Loss =================================
class dice_bce_loss(nn.Module):
    def __init__(self, batch = True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        
    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss
        
    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred, y_true)
        return a + b

    
# ================================== Dice Loss ===================================
class DiceCoeff(Function):

    # tensors are feeded in forward()
    def forward(self, input, target):
        self.save_for_backward(input, target) # ???
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
    
    def __call__(self, input, target):
        s = torch.FloatTensor(1).cuda().zero_() if input.is_cuda else torch.FloatTensor(1).zero_()
        for i, c in enumerate(zip(input, target)):
            s = s + (1-DiceCoeff().forward(c[0], c[1]))
        return s / (i + 1)
        
