# 0. Import

import numpy as np
#import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams["savefig.dpi"] = 600

# import custom library
from main_class import Counter

# load image and initialize Counter object
path = "c_12.jpg"
counter = Counter(image_path=path)

counter.detect_area_by_canny(radius=280)

# Crop samplea area in a circle shape.
counter.crop_samples(shrinkage_ratio=0.9)
counter.plot_cropped_samples()

counter.plot_cropped_samples(inverse=True)


counter.detect_colonies(min_size=7, max_size=15, threshold=0.1, verbose=True)

counter.plot_detected_colonies(plot="raw_inversed", col_num=6, save="results_inversed")






