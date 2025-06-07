#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:19:03 2025

@author: hsinyihung
"""


import os, glob
import imageio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


file_name = 'singal_data.csv'
directory = '/Users/hsinyihung/Desktop/'
fname = directory+file_name

df_control = pd.read_csv(fname)

x=df_control['actual_turn0']

y = df_control['predict_turn0']

# Calculate the linear regression

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

sns.regplot(data=df_control, x="actual_turn0", y="predict_turn0")
plt.savefig('localization.svg')