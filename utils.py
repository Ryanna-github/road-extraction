import torch

# change to the plot form
# all 0-1 scale
def change_tensor_to_plot(ts, to_numpy = True):
    ts = ts.squeeze()
    if ts.ndim > 3 | ts.ndim <= 1:
        raise ValueError("Can't plot img with shape: {}".format(ts.shape))
    elif ts.ndim == 3:
        if torch.tensor(ts.shape)[0] == 3:
            res = ts.permute(1, 2, 0)
        elif torch.tensor(ts.shape)[2] == 3:
            res = ts
        else:
            raise ValueError("Invalid data with shape {} (squeezed)".format(ts.shape))
    else:
        res = torch.cat([ts.unsqueeze(dim = 0)]*3).permute(1, 2, 0)
    if to_numpy:
        return res.cpu().numpy()
    else:
        return res.permute(2, 0, 1)