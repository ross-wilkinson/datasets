#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 14:30:42 2021

@author: rosswilkinson

Task:
    10-sec bouts of non-seated cycling at 5 W/kg and 70 RPM 
    on cycling rollers and a trainer.
    
Condition key:
    1: ad libitum lean on rollers
    2: minimal lean on rollers (subjects asked to prevent lean)
    3: locked in a trainer

Measures:
    Range of Vertical CoM Displacement
    Range of Bicycle Lean
    ...and many more!
    
"""

import os
import pandas as pd
import numpy as np
from pymer4.models import Lmer
from violinPlot import violinPlot
import matplotlib.pyplot as plt
from scipy import stats

### Set directory paths
expPath = '/Users/rosswilkinson/Google Drive/projects/bicycle-lean-rollers';
datPath = expPath + '/data'
docPath = expPath + '/docs'
codPath = expPath + '/matlab'
resPath = expPath + '/results'

### Load dataset
os.chdir(resPath)
df = pd.read_csv('rockNoRockRollers.csv')

### Normalize subject values to condition 1 (ad-lib)
# for i in range(1,20):    
#     intercept = np.mean(df['power'][(df['subject']==i) & (df['condition']==1)])
#     df['power'][df['subject']==i] = df['power'][df['subject']==i] / intercept

# =============================================================================
# RANGE BICYCLE LEAN
# =============================================================================

### Fit Linear Mixed-Effects Model
model = Lmer('RBLA_deg ~ (1+condition|subject)', data=df)
mdf = model.fit(factors={"condition": ["1", "2", "3"]})
# mdf = model.fit()
model.summary()

### Post-hoc comparisons
# model.post_hoc(marginal_vars="condition", grouping_vars=None, p_adjust="Tukey")

### Create data array for plotting individual linear regression
data = model.fixef.copy()
data.condition2 = data['(Intercept)'] + data['condition2']
data.condition3 = data['(Intercept)'] + data['condition3']

data = data[['condition3', '(Intercept)', 'condition2']]

percDiff1 = model.fixef['condition3'] / model.fixef['(Intercept)'] * 100
percDiff2 = model.fixef['condition2'] / model.fixef['(Intercept)'] * 100
temp = data['condition2'] - data['condition3']
percDiff3 = temp / data['condition3'] * 100

### Post-hoc comparisons
# Minimal vs. ad-lib
mu = model.fixef['condition2'].mean()
sigma = model.fixef['condition2'].std()
ci95 = stats.norm.interval(0.95, loc=mu, scale=sigma/np.sqrt(len(data)))
es = mu / sigma
cdf = stats.norm.cdf(abs(es)) 

# Locked vs. ad-lib
mu = model.fixef['condition3'].mean()
sigma = model.fixef['condition3'].std()
ci95 = stats.norm.interval(0.95, loc=mu, scale=sigma/np.sqrt(len(data)))
es = mu / sigma
cdf = stats.norm.cdf(abs(es)) 

meas = data

plt.figure(figsize=(16, 6))
plt.rcParams.update({'font.size': 16})
ax1 = plt.subplot(1,4,1)
plt.plot(meas.T, '-o', linewidth=1)
# plt.plot([0,1,2],data.mean(),'-ko',linewidth=3, markersize=9)
ax1.set_frame_on(False)
# ax1.set_ylim(-30,15)
ax1.set_ylabel('Range Vertical CoM Displacement (m)')
ax1.set_xticklabels(['Locked','ad-lib','Minimal'])

# ax2 = plt.subplot(1,4,2)
# kdepdf, x, ci95, mu, sigma, es = violinPlot(percDiff1,'o',ax2)
# ax2.set_frame_on(False)
# ax2.set_ylim(-100,100)
# ax2.set_ylabel('% Difference')
# ax2.set_xticks([])
# ax2.set_xlabel('Locked vs. ad-lib')
# ax2.axhline(linewidth=0.5, color='k')

# ax3 = plt.subplot(1,4,3)
# kdepdf, x, ci95, mu, sigma, es = violinPlot(percDiff2,'o',ax3)
# ax3.set_frame_on(False)
# ax3.set_ylim(-100,100)
# ax3.set_ylabel('% Difference')
# ax3.set_xticks([])
# ax3.set_xlabel('Minimal vs. ad-lib')
# ax3.axhline(linewidth=0.5, color='k')

# ax4 = plt.subplot(1,4,4)
# kdepdf, x, ci95, mu, sigma, es = violinPlot(percDiff3,'o',ax4)
# ax4.set_frame_on(False)
# ax4.set_ylim(-100,100)
# ax4.set_ylabel('% Difference')
# ax4.set_xticks([])
# ax4.set_xlabel('Minimal vs. Locked')
# ax4.axhline(linewidth=0.5, color='k')

plt.tight_layout()

os.chdir(docPath)
plt.savefig('RVCOMD.png', dpi=900)



# =============================================================================
# RANGE VERTICAL COM DISPLACEMENT
# =============================================================================

### Fit Linear Mixed-Effects Model
model = Lmer('RVCOMD_m ~ (1+condition|subject)', data=df)
mdf = model.fit(factors={"condition": ["1", "2", "3"]})
# mdf = model.fit()
model.summary()

### Post-hoc comparisons
# model.post_hoc(marginal_vars="condition", grouping_vars=None, p_adjust="Tukey")

### Create data array for plotting individual linear regression
data = model.fixef.copy()
data.condition2 = data['(Intercept)'] + data['condition2']
data.condition3 = data['(Intercept)'] + data['condition3']

data = data[['condition3', '(Intercept)', 'condition2']]

percDiff1 = model.fixef['condition3'] / model.fixef['(Intercept)'] * 100
percDiff2 = model.fixef['condition2'] / model.fixef['(Intercept)'] * 100
temp = data['condition2'] - data['condition3']
percDiff3 = temp / data['condition3'] * 100

meas = data

plt.figure(figsize=(16, 6))
plt.rcParams.update({'font.size': 16})
ax1 = plt.subplot(1,4,1)
plt.plot(meas.T, '-o', linewidth=1)
# plt.plot([0,1,2],data.mean(),'-ko',linewidth=3, markersize=9)
ax1.set_frame_on(False)
# ax1.set_ylim(-30,15)
ax1.set_ylabel('Range Vertical CoM Displacement (m)')
ax1.set_xticklabels(['Locked','ad-lib','Minimal'])

# ax2 = plt.subplot(1,4,2)
# kdepdf, x, ci95, mu, sigma, es = violinPlot(percDiff1,'o',ax2)
# ax2.set_frame_on(False)
# ax2.set_ylim(-100,100)
# ax2.set_ylabel('% Difference')
# ax2.set_xticks([])
# ax2.set_xlabel('Locked vs. ad-lib')
# ax2.axhline(linewidth=0.5, color='k')

# ax3 = plt.subplot(1,4,3)
# kdepdf, x, ci95, mu, sigma, es = violinPlot(percDiff2,'o',ax3)
# ax3.set_frame_on(False)
# ax3.set_ylim(-100,100)
# ax3.set_ylabel('% Difference')
# ax3.set_xticks([])
# ax3.set_xlabel('Minimal vs. ad-lib')
# ax3.axhline(linewidth=0.5, color='k')

# ax4 = plt.subplot(1,4,4)
# kdepdf, x, ci95, mu, sigma, es = violinPlot(percDiff3,'o',ax4)
# ax4.set_frame_on(False)
# ax4.set_ylim(-100,100)
# ax4.set_ylabel('% Difference')
# ax4.set_xticks([])
# ax4.set_xlabel('Minimal vs. Locked')
# ax4.axhline(linewidth=0.5, color='k')

plt.tight_layout()

os.chdir(docPath)
plt.savefig('RVCOMD.png', dpi=900)





