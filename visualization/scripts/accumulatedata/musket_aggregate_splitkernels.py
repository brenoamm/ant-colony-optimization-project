import csv
import numpy as np
import pandas as pd

my_data = pd.read_csv('../raw_data/musket_75_5-12_splitkernel.dat', delimiter=';', header=None)
currentcity = 0
results = pd.DataFrame(np.zeros((36, 12)))
sums = pd.DataFrame(np.zeros((1, 7)))

splits = 7
counter = 0
counterants = 0
numberants= 0
lastnumberants = 1024
print my_data

lastcity = my_data.iloc[0][2]
for index, row in my_data.iterrows():
    currentcity = row[2]
    numberants = row[3]
    if (currentcity == 11):
        print sums / counter

    if currentcity != lastcity:
        for x in range(0, 5):
            print x
            results.iloc[(counterants*7)+x][lastcity-1] = sums[x] / counter
        for x in range(5, 7):
            print x
            results.iloc[(counterants * 7) + x][lastcity - 1] = sums[x] * 15 / counter
        lastcity = currentcity
        sum = 0
        sums = pd.DataFrame(np.zeros((1, 7)))
        counter = 0
        counterants = 0
        numberants = 0
        lastnumberants = row[3]
    elif numberants != lastnumberants:
        for x in range(0, 5):
            results.iloc[(counterants * 9) + x][lastcity - 1] = sums[x] / counter
        for x in range(5, 7):
            results.iloc[(counterants * 9) + x][lastcity - 1] = sums[x] * 15 / counter
        counterants = counterants + 1
        counter = 0
        sum = 0
        sums = pd.DataFrame(np.zeros((1, 7)))
        lastnumberants = row[3]
    for x in range(0, 7):
        sums[x] += row[4 + x]
    counter = counter + 1
results.to_csv('result_kernel_breno_accumulated15.csv')
