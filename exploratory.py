# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:53:08 2018

@author: Natalie Menato
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

passengers = pd.read_csv('data/train.csv')

passengers.columns
passengers.info()
passengers.describe()

passengers.isnull().sum()
#177 null ages, 687 null cabin, 2 null embarked

passengers.pivot_table('Survived', index = 'Pclass')
passengers.pivot_table('Survived', index = 'Sex')
passengers.pivot_table('Survived', index = 'Age')
passengers.pivot_table('Survived', index = 'SibSp')
passengers.pivot_table('Survived', index = 'Parch')
passengers.pivot_table('Survived', index = 'Ticket')
passengers.pivot_table('Survived', index = 'Fare')
passengers.pivot_table('Survived', index = 'Cabin')
passengers.pivot_table('Survived', index = 'Embarked')

#binning of age
plt.hist(passengers['Age'].dropna())
passengers['Age_binned'] = pd.cut(passengers['Age'], bins=[0,15,60,80])
passengers.groupby('Age_binned').size()
passengers.pivot_table('Survived', index = 'Age_binned')

#investigating SibSp
passengers.pivot_table('Survived', index = 'SibSp')
passengers.groupby('SibSp').size()
passengers['SibSp'] = passengers['SibSp'] > 0

#investigating Parch
passengers.pivot_table('Survived', index = 'Parch')
passengers.groupby('Parch').size()
passengers['Parch'] = passengers['Parch'] > 0

#investigating Fare
plt.hist(passengers['Fare'].dropna())
passengers['Fare_binned'] = pd.cut(passengers['Fare'], bins=[0,15,30,550])
passengers.groupby('Fare_binned').size()
passengers.pivot_table('Survived', index = 'Fare_binned')

#investigating Cabin
passengers['has_cabin'] = pd.notna(passengers['Cabin'])
passengers.groupby('has_cabin').size()
passengers.pivot_table('Survived', index = 'has_cabin')
passengers['Cabin_letter'] = passengers['Cabin'].str[0:1]
passengers.groupby('Cabin_letter').size()
passengers.pivot_table('Survived', index = 'Cabin_letter')

passengers[['Survived', 'Fare', 'Cabin_letter', 'Pclass']].dropna()
passengers.groupby(('Pclass', 'Cabin_letter')).size()

#investigating Embarked
passengers.pivot_table('Survived', index = 'Embarked')
passengers.groupby('Embarked').size()