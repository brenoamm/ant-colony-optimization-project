#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:58:45 2020

@author: bamm, Nina Herrmann
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ll_data = pd.read_csv('../../../data_aggregation/Knapsack/Knapsack_LL_Quadro.csv', delimiter=';', header=None)
hl_data = pd.read_csv('../../../data_aggregation/Knapsack/Knapsack_HL_Quadro.csv', delimiter=';', header=None)
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

#iterate over files
for f in range(6):
    for x in range(4):
        HL_Average[x][f] = hl_data[4][x + (f*4)]
        LL_Average[x][f] = ll_data[4][x + (f*4)]

#define size
print HL_Average
print LL_Average
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 10))

# X-Achsis
Knapsacks = [7,6,5,4,3,1]
ind = np.arange(len(Knapsacks))
width = 0.35         # the width of the bars

BrenoBars_1024 = axes[0, 0].bar(ind, ll1024Average, width, bottom=0, color = '#A60628')
BrenoBars_2048 = axes[1, 0].bar(ind, ll2048Average, width, bottom=0, color = '#A60628')
BrenoBars_4096 = axes[0, 1].bar(ind, ll4096Average, width, bottom=0, color = '#A60628')
BrenoBars_8192 = axes[1, 1].bar(ind, ll8192Average, width, bottom=0, color = '#A60628')

MusketBars_1024 = axes[0, 0].bar(ind+width, Musket1024Average, width,  bottom=0, color = '#348ABD')
MusketBars_2048 = axes[1, 0].bar(ind+width, Musket2048Average, width,  bottom=0, color = '#348ABD')
MusketBars_4096 = axes[0, 1].bar(ind+width, Musket4096Average, width,  bottom=0, color = '#348ABD')
MusketBars_8192 = axes[1, 1].bar(ind+width, Musket8192Average, width,  bottom=0, color = '#348ABD')

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

axes[0, 0].set_xticks(ind + width / 2)
axes[0, 0].set_xticklabels(Knapsacks)
for ax in axes.flat:
    ax.label_outer()
for ax in fig.get_axes():
    ax.label_outer()
axes[0, 0].legend((BrenoBars_1024[0], MusketBars_1024[0]), ('Low-Level', 'Musket'))
axes[0, 0].autoscale_view()

plt.show()
