import csv
import numpy as np
import pandas as pd

my_data = pd.read_csv('../raw_data/lowlevel_1,3,5-12_splitkernel.csv', delimiter=',', header=None)
currentcity = 0
results = pd.DataFrame(np.zeros((32, 12)))
sums = pd.DataFrame(np.zeros((1, 9)))

splits = 9
counter = 0
counterants = 0
numberants= 0
lastnumberants = 1024
lastcity = my_data.iloc[0][2]
print lastcity
for index, row in my_data.iterrows():
    currentcity = row[2]
    numberants = row[3]
    if currentcity != lastcity:
        for x in range(0, 8):
            results.iloc[(counterants*8)+x][lastcity-1] = sums[x] / counter
        lastcity = currentcity
        sum = 0
        sums = pd.DataFrame(np.zeros((1, 9)))
        counter = 0
        counterants = 0
        numberants = 0
        lastnumberants = row[3]
    elif numberants != lastnumberants:
        for x in range(0, 8):
            #print sums[x]
            results.iloc[(counterants * 8) + x][lastcity - 1] = sums[x] / counter
        counterants = counterants + 1
        counter = 0
        sum = 0
        sums = pd.DataFrame(np.zeros((1, 9)))
        lastnumberants = row[3]
    for x in range(0, 8):
        sums[x] += row[4 + x]
    counter = counter + 1
results.to_csv('ll_breno.csv')
