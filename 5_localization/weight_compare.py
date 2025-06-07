#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:54:32 2025

@author: hsinyihung
"""

import os, glob
import imageio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

control_name = 'D.Melano_weight_webvibration.csv'

directory = '/Users/hsinyihung/Documents/PhD/JHU/GordusLab/Spider_prey_vibration/web_vibration/result/'
df = pd.read_csv(directory + control_name, header=0)

# Calculate mean and SEM
means = df.mean()
sems = df.sem()  # pandas has a built-in method for SEM!

# Create scatter plot with error bars (SEM)
plt.figure(figsize=(8, 6))
x = np.arange(len(df.columns))
plt.errorbar(x, means, yerr=sems, fmt='o', capsize=5, linestyle='None', color='black', ecolor='black')

# Plot individual data points with jitter
for i, col in enumerate(df.columns):
    jitter = np.random.normal(0, 0.05, size=len(df[col]))  # small horizontal jitter
    if col == 'V weight ':
        plt.scatter(np.full_like(df[col], x[i]) + jitter, df[col], alpha=0.6, label=f'{col} data' if i == 0 else "",
                    color='limegreen')
    elif col == 'M weight ':
        plt.scatter(np.full_like(df[col], x[i]) + jitter, df[col], alpha=0.6, label=f'{col} data' if i == 0 else "",
                    color='magenta')
    else:
        plt.scatter(np.full_like(df[col], x[i]) + jitter, df[col], alpha=0.6, label=f'{col} data' if i == 0 else "",
                    color='darkgrey')

# Customize plot
plt.xticks(x, df.columns)
plt.ylabel('Weight (g)')
plt.title('Mean ± SEM for Each Column')

plt.tight_layout()
plt.savefig('weight_compare.svg')

from scipy.stats import mannwhitneyu

# Compare column 'A' and 'B'
stat, p = mannwhitneyu(df['V weight '], df['M weight '], alternative='two-sided')

print(f'Mann–Whitney U statistic = {stat:.3f}, p-value = {p:.4f}')
