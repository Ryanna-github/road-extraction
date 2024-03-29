import torch
import cv2
import numpy as np
import time
import math

# radiometric calibration for df new data
# img.shape = 3*x*x
# return .shape = 3*x*x
def radio_calibrate(img):
    I=2003
    J=2
    K=20
    JD=K-32075+1461*(I+4800+ (J-14)/12)/4+367*(J-2-(J-14)/12*12)/12-3*((I+4900+(J-14)/12)/100)/4
    ESUNI71=196.9
    D=1-0.01674*math.cos((0.9856*(JD-4)*math.pi/180))
    cos=math.cos(math.radians(90-41.3509605))
    inter=(math.pi*D*D)/(ESUNI71*cos*cos)
    Lmini=-6.2
    Lmax=293.7
    Qcal=1
    Qmax=255
    LIMIN=Lmini+(Qcal*(Lmax-Lmini)/Qmax)
    LI=(0.01*ESUNI71*cos*cos)/(math.pi*D*D)
    Lhazel=LIMIN-LI
    
    def copy(img,new1):
        new1= np.zeros(img.shape,dtype='uint16')
        new1[:,:] = img[:,:]

    def computL(gain,Dn,bias):
        return (gain*Dn+bias)

    result=np.zeros(img.shape,dtype='uint16')
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            Lsat=computL(1.18070871,img[i,j],-7.38070852)
            result[i,j]=inter*(Lsat-Lhazel)*1000
    rmin = result.min()
    rmax = result.max()
    result2 = (result - rmin)/(rmax - rmin)
    return result2

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



class Timer(object):
    """ Simple timer class for measuring time consuming """

    def __init__(self):
        self._start_time = 0.0
        self._end_time = 0.0
        self._elapsed_time = 0.0
        self._is_running = False

    def start(self):
        self._is_running = True
        self._start_time = time.time()

    def restart(self):
        self.start()

    def stop(self):
        self._is_running = False
        self._end_time = time.time()

    def elapsed_time(self):
        self._end_time = time.time()
        self._elapsed_time = self._end_time - self._start_time
        if not self.is_running:
            return 0.0

        return self._elapsed_time

    @property
    def is_running(self):
        return self._is_running


def calculate_eta(remaining_step, speed):
    if remaining_step < 0:
        remaining_step = 0
    remaining_time = int(remaining_step * speed)
    result = "{:0>2}:{:0>2}:{:0>2}"
    arr = []
    for i in range(2, -1, -1):
        arr.append(int(remaining_time / 60**i))
        remaining_time %= 60**i
    return result.format(*arr)

class Counter:
    def __init__(self):
        self.data = dict()

    def append(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []

            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()

            self.data[key].append(value)

    def __getattr__(self, key):
        if key not in self.data:
            return 0
        return np.mean(self.data[key])