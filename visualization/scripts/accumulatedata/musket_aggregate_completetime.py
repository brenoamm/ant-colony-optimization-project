import csv

import numpy as np
import pandas as pd
from decimal import Decimal
my_data=pd.read_csv('../data_aggregation/BPP70short.csv', delimiter=';', header=None)
currentcity = 0
results = pd.DataFrame(np.zeros((4, 5)))
sum = 0
counter = 0
counterants = 0
numberants= 0
lastnumberants = 1024
print my_data
lastcity = 0
for index, row in my_data.iterrows():

    currentcity = row[1]
    numberants = row[0]
    if currentcity != lastcity:
        print lastcity
        results.iloc[counterants][lastcity] = sum / counter
        lastcity = currentcity
        sum = 0
        counter = 0
        counterants = 0
        numberants = 0
        lastnumberants = row[0]
    elif numberants != lastnumberants:
        results.iloc[counterants][lastcity] = sum / counter
        counterants = counterants + 1
        counter = 0
        sum = 0
        lastnumberants = row[0]
    afloatnumber = float(row[4])
    sum += afloatnumber
    counter = counter + 1
print results
results.to_csv('result_BPP70_musket.csv')
