from frame.config import *

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def forward(self, input, target, laplace = 1):
        inter = torch.dot(input.view(-1), target.view(-1))
        union = torch.sum(input) + torch.sum(target)
        loss = (2 * inter.float() + laplace) / (union.float() + laplace)
        return loss
        
        