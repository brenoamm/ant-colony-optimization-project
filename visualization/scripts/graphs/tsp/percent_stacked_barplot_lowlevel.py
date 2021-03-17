# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
 
md = pd.read_csv('../../../data_aggregation/TSP/LowLevel/LowLevel_1,3,5-12_splitkernels.csv', delimiter=',', header=None).T

#define size
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 10))
fig.subplots_adjust(hspace=0.15, wspace=0.05)
fig.suptitle('Kernel Percentage - Low Level ', fontsize=16, y=0.95)

legendlabels = ["" for x in range(10)]
#legendlabels = pd.DataFrame(np.zeros((1, 10)))
legendlabels[0] = 'Initialize Datastructures'
legendlabels[1] = 'Read Data and Copy to Device'
legendlabels[2] = 'Calculate Distance'
legendlabels[3] = 'Calculate Iroueltte'
legendlabels[4] = 'Route Kernel'
legendlabels[5] = 'Update Best Sequence'
legendlabels[6] = 'Update Pheromones'
legendlabels[7] = 'Copy Data'

axes[0, 0].set_title('1024 Ants')
axes[1, 0].set_title('2048 Ants')
axes[0, 1].set_title('4096 Ants')
axes[1, 1].set_title('8192 Ants')

axes[0, 0].set_ylabel('%')
axes[1, 0].set_ylabel('%')
axes[0, 1].set_ylabel('%')
axes[1, 1].set_ylabel('%')

axes[1, 0].set_xlabel('tsp instance')
axes[1, 1].set_xlabel('tsp instance')
 
# From raw value to percentage

setup_index = 0
barWidth = 0.85
colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
pos = np.arange(10)

for x_range in range(2):
    for y_range in range(2):    
        totals = np.array([i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(md[1+(8*setup_index)],md[2+(8*setup_index)],md[3+(8*setup_index)],md[4+(8*setup_index)],md[5+(8*setup_index)],md[6+(8*setup_index)],md[7+(8*setup_index)],md[8+(8*setup_index)])])
        percents = np.zeros((8, 10))

        for x in range(8):
            percents[x] = np.array([i / j * 100 for i,j in zip(md[x+1+(8*setup_index)], totals)])

        bottoms = np.array([0,0,0,0,0,0,0,0,0,0])
      
        for x in range(8):
            # Create green Bars
            a =  percents[x]
            axes[x_range, y_range].bar(pos, a, color=colors[x], bottom=bottoms, edgecolor='white', width=barWidth, label=legendlabels[x])
            bottoms = bottoms + percents[x]
        
        setup_index = setup_index +1

width = 0.45
axes[0, 0].set_xticks((pos+ (width / 2))-0.15)
axes[0, 0].set_xticklabels(['dj38','qa194','d198','lin318','pcb442','rat783','pr1002','pcb1173','d1291','pr2392'])

# Put a legend below current axis
axes[0,0].legend(loc='upper center', bbox_to_anchor=(1.05, -1.30),fancybox=True, shadow=True, ncol=5)

for ax in axes.flat:
    ax.label_outer()
for ax in fig.get_axes():
    ax.label_outer()
# Show graphic
plt.show()