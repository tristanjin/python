'''
Created on 2018. 11. 6.

@author: tristanjin
'''
from __future__ import print_function

import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

DJI = pd.read_csv("./data/_DJI.csv")
CPIAUCSL = pd.read_csv("./data/CPIAUCSL.csv")
DGS10 = pd.read_csv("./data/DGS10.csv")
fed = pd.read_csv("./data/fed-funds-rate-historical-chart.csv")
LNS = pd.read_csv("./data/LNS14000024.csv")

merged_data=pd.merge(DJI,CPIAUCSL, on="DATE", how="left")
merged_data=pd.merge(merged_data,DGS10, on="DATE", how="left")
merged_data=pd.merge(merged_data,fed, on="DATE", how="left")
merged_data=pd.merge(merged_data,LNS, on="DATE", how="left")
#merged_data=merged_data.replace(np.nan,0)
#merged_data=merged_data.replace(".",0)

merged_data_tensor=tf.convert_to_tensor(merged_data)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
merged_data_tensor=sess.run(merged_data_tensor.replace(np.nan,0))


print(merged_data_tensor)
