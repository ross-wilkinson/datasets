#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 14:30:42 2021

@author: rosswilkinson

Condition codes:
Posture 1: Seated
Posture 2: Standing
Grip 1: Grip (Normal)
Grip 2: No Grip (Resting fists on top of handlebar)
"""

import pandas as pd
from pymer4.models import Lmer

### Load dataset
df = pd.read_csv('https://github.com/ross-wilkinson/datasets/blob/main/gripNoGrip/gripNoGrip.csv')

### Fit Linear Mixed-Effects Model
model = Lmer('MaxPowerCycle ~ Posture + Grip + (1+Posture|Subject) + (1+Grip|Subject)', data=df)
mdf = model.fit()
model.summary()

### Plot model parameters
model.plot_summary()
model.plot('Posture')
model.plot('Grip')

