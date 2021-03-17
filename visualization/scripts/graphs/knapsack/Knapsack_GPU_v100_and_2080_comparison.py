#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:58:45 2020

@author: bamm, Nina Herrmann
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ll_data = pd.read_csv('../../../data_aggregation/Knapsack/Knapsack_LL_v100.csv', delimiter=';', header=None)
hl_data = pd.read_csv('../../../data_aggregation/Knapsack/Knapsack_HL_v100.csv', delimiter=';', header=None)

ll2080_data = pd.read_csv('../../../data_aggregation/Knapsack/Knapsack_LL_2080.csv', delimiter=';', header=None)
hl2080_data = pd.read_csv('../../../data_aggregation/Knapsack/Knapsack_HL_2080.csv', delimiter=';', header=None)


#extract data from files
Musket1024Average = [0,0,0,0,0,0]
Musket2048Average = [0,0,0,0,0,0]
Musket4096Average = [0,0,0,0,0,0]
Musket8192Average = [0,0,0,0,0,0]

HL_Average = [Musket1024Average, Musket2048Average, Musket4096Average, Musket8192Average]

ll1024Average = [0,0,0,0,0,0]
ll2048Average = [0,0,0,0,0,0]
ll4096Average = [0,0,0,0,0,0]
ll8192Average = [0,0,0,0,0,0]

LL_Average = [ll1024Average, ll2048Average, ll4096Average, ll8192Average]

#extract data from files
Musket1024Average2080 = [0,0,0,0,0,0]
Musket2048Average2080 = [0,0,0,0,0,0]
Musket4096Average2080 = [0,0,0,0,0,0]
Musket8192Average2080 = [0,0,0,0,0,0]

HL_Average2080 = [Musket1024Average2080, Musket2048Average2080, Musket4096Average2080, Musket8192Average2080]

ll1024Average2080 = [0,0,0,0,0,0]
ll2048Average2080 = [0,0,0,0,0,0]
ll4096Average2080 = [0,0,0,0,0,0]
ll8192Average2080 = [0,0,0,0,0,0]

LL_Average2080 = [ll1024Average2080, ll2048Average2080, ll4096Average2080, ll8192Average2080]

#iterate over files
for f in range(6):
    for x in range(4):
        HL_Average[x][f] = hl_data[4][x + (f*4)]
        LL_Average[x][f] = ll_data[4][x + (f*4)]
        HL_Average2080[x][f] = hl2080_data[4][x + (f*4)]
        LL_Average2080[x][f] = ll2080_data[3][x + (f*4)]

#define size
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 10))

# X-Achsis
Knapsacks = [1,3,4,5,6,7]
ind = np.arange(len(Knapsacks))
width = 0.17         # the width of the bars

BrenoBars_1024 = axes[0, 0].bar(ind, ll1024Average, width, bottom=0, color = '#A60628')
BrenoBars_2048 = axes[1, 0].bar(ind, ll2048Average, width, bottom=0, color = '#A60628')
BrenoBars_4096 = axes[0, 1].bar(ind, ll4096Average, width, bottom=0, color = '#A60628')
BrenoBars_8192 = axes[1, 1].bar(ind, ll8192Average, width, bottom=0, color = '#A60628')

MusketBars_1024 = axes[0, 0].bar(ind+width, Musket1024Average, width,  bottom=0, color = '#348ABD')
MusketBars_2048 = axes[1, 0].bar(ind+width, Musket2048Average, width,  bottom=0, color = '#348ABD')
MusketBars_4096 = axes[0, 1].bar(ind+width, Musket4096Average, width,  bottom=0, color = '#348ABD')
MusketBars_8192 = axes[1, 1].bar(ind+width, Musket8192Average, width,  bottom=0, color = '#348ABD')
                      
BrenoBars_1024_2080 = axes[0, 0].bar(ind+(2*width), ll1024Average2080, width, bottom=0, color = '#488f31')
BrenoBars_2048_2080 = axes[1, 0].bar(ind+(2*width), ll2048Average2080, width, bottom=0, color = '#488f31')
BrenoBars_4096_2080 = axes[0, 1].bar(ind+(2*width), ll4096Average2080, width, bottom=0, color = '#488f31')
BrenoBars_8192_2080 = axes[1, 1].bar(ind+(2*width), ll8192Average2080, width, bottom=0, color = '#488f31')

MusketBars_1024_2080 = axes[0, 0].bar(ind+(3*width), Musket1024Average2080, width,  bottom=0, color = '#beae6b')
MusketBars_2048_2080 = axes[1, 0].bar(ind+(3*width), Musket2048Average2080, width,  bottom=0, color = '#beae6b')
MusketBars_4096_2080 = axes[0, 1].bar(ind+(3*width), Musket4096Average2080, width,  bottom=0, color = '#beae6b')
MusketBars_8192_2080 = axes[1, 1].bar(ind+(3*width), Musket8192Average2080, width,  bottom=0, color = '#beae6b')

axes[0, 0].set_title('1024 Ants')
axes[1, 0].set_title('2048 Ants')
axes[0, 1].set_title('4096 Ants')
axes[1, 1].set_title('8192 Ants')

axes[0, 0].set_ylabel('seconds')
axes[1, 0].set_ylabel('seconds')
axes[0, 1].set_ylabel('seconds')
axes[1, 1].set_ylabel('seconds')

axes[0, 0].set_xlabel('problem')
axes[1, 0].set_xlabel('problem')
axes[0, 1].set_xlabel('problem')
axes[1, 1].set_xlabel('problem')

axes[0, 0].set_xticks(ind + 1.5*width)
axes[0, 0].set_xticklabels(Knapsacks)
#for ax in axes.flat:
#    ax.label_outer()
#for ax in fig.get_axes():
#    ax.label_outer()
axes[0, 0].legend((BrenoBars_1024[0], MusketBars_1024[0],BrenoBars_1024_2080[0], MusketBars_1024_2080[0]), ('v100 Low-Level', 'v100 Musket','2080 Ti Low-Level', '2080 Ti  Musket'))
axes[0, 0].autoscale_view()

plt.show()
