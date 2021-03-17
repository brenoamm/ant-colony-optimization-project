import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

my_data = pd.read_csv('../../../raw_data/BPP/HighLevel/musk_packing_3.dat', delimiter=';', header=None)
breno_data = pd.read_csv('../../../data_aggregation/BPP/LowLevel/ll_packing_3.dat', delimiter=';', header=None)

#define size
#fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 10))

#fig.suptitle('BPP - Packing Kernel Runtime Comparison', fontsize=16)

# X-Achsis
ind = 0    # the x locations for the groups
width = 0.35         # the width of the bars

MusketAverage = [0,0,0,0]
lowlevelAverage = [0,0,0,0]

for x in range(4):
    ma = 0
    lla = 0
    
    for y in range(10):
        ma = ma + my_data.at[((x*10)+y), 4]
        lla = lla + breno_data.at[((x*10)+y), 4]
    
    ma = ma/10
    lla = lla/10
    MusketAverage[x] = ma
    lowlevelAverage[x] = lla

# Data from Musket
Musket1024 = MusketAverage[0]
Musket2048 = MusketAverage[1]
Musket4096 = MusketAverage[2]
Musket8192 = MusketAverage[3]

# Data Breno
Breno1024 = lowlevelAverage[0]
Breno2048 = lowlevelAverage[1]
Breno4096 = lowlevelAverage[2]
Breno8192 = lowlevelAverage[3]

# width of the bars
barWidth = 0.3

# The x position of bars
r1 = np.arange(len(MusketAverage))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(9, 3))

# Create blue bars
plt.bar(r1, lowlevelAverage, width = barWidth, color = '#A60628', edgecolor = 'black', capsize=7, label='Low-Level')
plt.bar(r2, MusketAverage, width = barWidth, color = '#348ABD', edgecolor = 'black', capsize=7, label='Musket')

# general layout
plt.xticks([r + barWidth/2 for r in range(len(lowlevelAverage))], ['1024', '2048', '4096', '8192'])
plt.title('Packing Kernel Runtime Comparison - problem 3')
plt.ylabel('seconds')
plt.xlabel('ants')
plt.legend()

plt.show()
