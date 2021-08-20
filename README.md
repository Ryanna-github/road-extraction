# File Structure

- cache/ temporary saved middle data.
- checkpoints/ checkpoints generated while running, some of them would be saved to models/
- experiments/ files used for trials, whose file name usually begin with experiment start-date.
- frame/ the main module.
    - \_\_init\_\_.py
    - evaluate.py: evaluate the model performance.
    - pipeline.py: integration of data processsing, training and testing.
    - utils.py: other handy tools.
    - visualize.py: visualize the results.
- logs/ log data.
- models/ model class definition module.
    - \_\_init\_\_.py
    - unet.py
    - linknet.py
    - ...
- runs/ tensorboard recording files.