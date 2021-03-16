import torch
import cv2
import numpy as np

# change to the plot form
# all 0-1 scale
# return: 
#     if to_numpy: (d, d, 3) np.ndarray variable
#     else: (3, d, d) torch.Tensor
def change_tensor_to_plot(ts, to_numpy = True):
    ts = ts.squeeze()
    # if ts is a numpy array ...
    if isinstance(ts, np.ndarray):
        if ts.ndim == 2:
            return np.concatenate([np.expand_dims(ts, axis = 0)]*3).transpose(1, 2, 0) if to_numpy else torch.tensor( np.concatenate([np.expand_dims(ts, axis = 0)]*3))
        elif ts.shape[0] == 3:
            return ts.transpose(1, 2, 0) if to_numpy else torch.tensor(ts)
        elif ts.shape[2] == 3:
            return ts if to_numpy else torch.tensor(ts).permute(2, 0, 1)
        else:
            raise ValueError("Wrong np.ndarray shape {}".format(ts.shape))
    # if ts is a torch tensor
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

# improve test results of model
def get_improved_result(ts, threshold = 500):
    ts = change_tensor_to_plot(ts, to_numpy = True)
    ret, img = cv2.threshold(np.array(ts[:,:,0]), 0.5, 255, cv2.THRESH_BINARY) # 二值化
    contours, _ = cv2.findContours(img.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= threshold:
            cv_contours.append(contour)
    cv2.fillPoly(img, cv_contours, (0, 0, 0))
    img = change_tensor_to_plot(img, to_numpy = True)
    return img