import csv

import numpy as np
import pandas as pd
from decimal import Decimal
my_data=pd.read_csv('../raw_data/splitkernel70.csv', delimiter=';')
currentcity = 0
results = pd.DataFrame(np.zeros((4, 12)))
sum = 0
counter = 0
counterants = 0
numberants= 0
lastnumberants = 1024
print my_data
lastcity = 1
for index, row in my_data.iterrows():
    currentcity = row[2]
    numberants = row[3]
    if currentcity == 2:
        break
    if currentcity == 4:
        break
    if currentcity != lastcity:
        print row
        results.iloc[counterants][lastcity] = sum / counter
        lastcity = currentcity
        sum = 0
        counter = 0
        counterants = 0
        numberants = 0
        lastnumberants = row[3]
    elif numberants != lastnumberants:
        results.iloc[counterants][lastcity] = sum / counter
        counterants = counterants + 1
        counter = 0
        sum = 0
        lastnumberants = row[3]
    afloatnumber = float(row[13])
    sum += afloatnumber
    counter = counter + 1
print results
results.to_csv('result_TSP70_musket.csv')
