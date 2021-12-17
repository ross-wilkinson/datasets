#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 14:30:42 2021

@author: rosswilkinson

Task:
    Ergometer cycling for 10 sec at an individualized constant power output.
    Equal to 50% of the subject's maximal instantaneous sprint power.
    
Condition key:
    Posture 1: Seated
    Posture 2: Standing
    Cadence 1: 70 RPM
    Cadence 2: 120 RPM

Measures:
    Mean power at hip, knee, and ankle over each crank cycle.
    Normalized to the subject's body mass.
    
"""

import pandas as pd
import numpy as np
from pymer4.models import Lmer

### Load dataset
df = pd.read_csv('sitStand.csv')

### Normalize subject values to condition 1 (ad-lib)
for i in range(len(df)):       
    df.iloc[i,3:] = df.iloc[i,3:] / df['CrankPower'][i]

# =============================================================================
# HIP
# =============================================================================

### Fit Linear Mixed-Effects Model
modelHip = Lmer('HipPower ~ Posture * Cadence + (1+Posture*Cadence|Subject)', data=df)
mdfHip = modelHip.fit(factors={"Posture": ["1", "2"], "Cadence": ["1", "2"]})
modelHip.summary()

### Post-hoc comparisons
modelHip.post_hoc(marginal_vars="Posture", 
                  grouping_vars="Cadence", 
                  p_adjust="tukey")

modelHip.post_hoc(marginal_vars="Cadence", 
                  grouping_vars="Posture", 
                  p_adjust="tukey")

### Plot model parameters
#modelHip.plot_summary()
#modelHip.plot('Posture')
#modelHip.plot('Cadence')

# =============================================================================
# KNEE
# =============================================================================

### Fit Linear Mixed-Effects Model
modelKnee = Lmer('KneePower ~ Posture * Cadence + (1+Posture*Cadence|Subject)', data=df)
mdfKnee = modelKnee.fit(factors={"Posture": ["1", "2"], "Cadence": ["1", "2"]})
modelKnee.summary()

### Post-hoc comparisons
modelKnee.post_hoc(marginal_vars="Posture", 
                  grouping_vars="Cadence", 
                  p_adjust="tukey")

modelKnee.post_hoc(marginal_vars="Cadence", 
                  grouping_vars="Posture", 
                  p_adjust="tukey")

### Plot model parameters
#modelKnee.plot_summary()
#modelKnee.plot('Posture')
#modelKnee.plot('Cadence')

# =============================================================================
# ANKLE
# =============================================================================

### Fit Linear Mixed-Effects Model
modelAnkle = Lmer('AnklePower ~ Posture * Cadence + (1+Posture*Cadence|Subject)', data=df)
mdfAnkle = modelAnkle.fit(factors={"Posture": ["1", "2"], "Cadence": ["1", "2"]})
modelAnkle.summary()

### Post-hoc comparisons
modelAnkle.post_hoc(marginal_vars="Posture", 
                  grouping_vars="Cadence", 
                  p_adjust="tukey")

modelAnkle.post_hoc(marginal_vars="Cadence", 
                  grouping_vars="Posture", 
                  p_adjust="tukey")

### Plot model parameters
#modelAnkle.plot_summary()
#modelAnkle.plot('Posture')
#modelAnkle.plot('Cadence')






