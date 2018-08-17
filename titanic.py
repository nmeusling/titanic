# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:58:59 2018

@author: Natalie Menato
"""

import pandas as pd

def create_features(passengers):
    #binning and creation of features
    passengers['Age_binned'] = pd.cut(passengers['Age'], bins=[0,15,60,80])
    passengers['SibSp'] = passengers['SibSp'] > 0
    passengers['Parch'] = passengers['Parch'] > 0
    passengers['Fare_binned'] = pd.cut(passengers['Fare'], bins=[0,15,30,550])
    passengers['has_cabin'] = pd.notna(passengers['Cabin'])
    passengers['Cabin_letter'] = passengers['Cabin'].str[0:1]
    return passengers

    
