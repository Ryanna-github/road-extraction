import sys
sys.path.insert(0, "../") # Avoid naming conflicts with system files
import frame.dataloader
import frame.pipeline
import frame.evaluate
from frame.config import *


# The global variable
classes = ['background', 'road']
colormap = [[0 , 0, 0], [255, 255, 255]]