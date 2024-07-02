# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:18:41 2024

@author: AdamPezzutti
"""

import pandas as pd
df = pd.read_csv(r'OnlineNewsPopularity.csv')

print(df.info())
print(df.head())
print(df.describe())


"""
Business Goal: Predirre la popolarità mediatica di 
vari articoli di Mashable,
prendendo in considerazione diverse metriche derivanti da
una indagine statistica.
"""