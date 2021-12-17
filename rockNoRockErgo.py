#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 14:30:42 2021

@author: rosswilkinson

Task:
    All-out 5-sec sprints in a non-seated posture on a modified cycling ergometer
    that could lean from from side to side or be locked to prevent lean.
    
Condition key:
    1: ad libitum lean
    2: minimal lean (subjects asked to prevent lean)
    3: locked

Measures:
    Maximal 1-s crank power. 
    Cadence and torque at max. power. 
    Range of ergometer lean.
    
"""

import pandas as pd
import numpy as np
from pymer4.models import Lmer

### Load dataset
df = pd.read_csv('rockNoRockErgo.csv')

### Normalize subject values to condition 1 (ad-lib)
for i in range(1,20):    
    intercept = np.mean(df['power'][(df['subject']==i) & (df['condition']==1)])
    df['power'][df['subject']==i] = df['power'][df['subject']==i] / intercept

# =============================================================================
# POWER
# =============================================================================

### Fit Linear Mixed-Effects Model
model = Lmer('power ~ condition + (1+condition|subject)', data=df)
mdf = model.fit(factors={"condition": ["1", "2", "3"]})
#model.summary()

### Post-hoc comparisons
model.post_hoc(marginal_vars="condition", grouping_vars=None, p_adjust="Tukey")

### Plot model parameters
#model.plot_summary()
#modelHip.plot('Posture')
#modelHip.plot('Cadence')

# =============================================================================
# KNEE
# =============================================================================

### Fit Linear Mixed-Effects Model
modelKnee = Lmer('KneePower ~ Posture * Cadence + (1+Posture|Subject) + (1+Cadence|Subject)', data=df)
mdfKnee = modelKnee.fit()
modelKnee.summary()

### Plot model parameters
#modelKnee.plot_summary()
#modelKnee.plot('Posture')
#modelKnee.plot('Cadence')

# =============================================================================
# ANKLE
# =============================================================================

### Fit Linear Mixed-Effects Model
modelAnkle = Lmer('AnklePower ~ Posture * Cadence + (1+Posture|Subject) + (1+Cadence|Subject)', data=df)
mdfAnkle = modelAnkle.fit()
modelAnkle.summary()

### Plot model parameters
#modelAnkle.plot_summary()
#modelAnkle.plot('Posture')
#modelAnkle.plot('Cadence')






