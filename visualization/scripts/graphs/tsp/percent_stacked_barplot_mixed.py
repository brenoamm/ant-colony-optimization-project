# libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

md = pd.read_csv('../../../data_aggregation/TSP/LowLevel/LowLevel_1,3,5-12_splitkernels.csv', delimiter=',', header=None).T
md_musket = pd.read_csv('../../../data_aggregation/TSP/HighLevel/Musket_1,3,5-12_kernel_sumofiterations.csv', delimiter=',', header=None).T
matplotlib.style.use('default')

# define size
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 10))
fig.subplots_adjust(hspace=0.15, wspace=0.1)
# fig.suptitle('Kernel Percentage', fontsize=16, y=0.95)
axes[0].grid(color='gray', which='both')
axes[1].grid(color='gray')
axes[0].set_axisbelow(True)
axes[1].set_axisbelow(True)


legendlabels = ["" for x in range(2)]
legendlabels[0] = 'Calculate Route'
legendlabels[1] = 'Other Calculations'

axes[0].set_title('Low Level')
axes[1].set_title('Musket')
axes[1].set_ylabel('%')
axes[0].set_ylabel('%')

setup_index = 0
barWidth = 0.65
colors = np.array(['#A60628', '#f95d7f'])
pos = np.arange(10)

percents_ll = np.zeros((2, 10))

othertimes = np.array(md[1 + (8 * setup_index)] + md[2 + (8 * setup_index)] + md[3 + (8 * setup_index)] +
                       md[4 + (8 * setup_index)] + md[6 + (8 * setup_index)] +
                       md[7 + (8 * setup_index)] + md[8 + (8 * setup_index)])
percents_ll[0] = np.array((md[5 + (8 * setup_index)] / (othertimes + md[5 + (8 * setup_index)])) * 100)
percents_ll[1] = np.array((othertimes / (othertimes + md[5 + (8 * setup_index)])) * 100)
bottoms = np.array([0,0,0,0,0,0,0,0,0,0])
for x in range(2):
    # Create green Bars
    a = percents_ll[x]
    axes[0].bar(pos, a, color=colors[x], bottom=bottoms, edgecolor='white', width=barWidth, label=legendlabels[x])
    bottoms = bottoms + percents_ll[x]

colors = np.array(['#348ABD', '#8bc0df'])

# Start Musket plot
percents = np.zeros((2, 10))

othertimes = np.array(md_musket[1 + (9 * setup_index)] + md_musket[2 + (9 * setup_index)] + md_musket[3 + (9 * setup_index)] +
                       md_musket[4 + (9 * setup_index)] + md_musket[6 + (9 * setup_index)] + md_musket[7 + (9 * setup_index)] +
                       md_musket[8 + (9 * setup_index)] + md_musket[9 + (9 * setup_index)])
routetime = np.array(md_musket[5 + (9 * setup_index)])
percents[0] = np.array((routetime / (othertimes + routetime)) * 100)
percents[1] = np.array((othertimes / (othertimes + routetime)) * 100)
bottoms = np.array([0,0,0,0,0,0,0,0,0,0])
plt.subplots_adjust(left=0.2, bottom=0.3)
cell_text = []
cell_text.append(['%1.1f' % (x) for x in percents[0]])
cell_text.append(['%1.1f' % (x) for x in percents[1]])
# Add a table at the bottom of the axes
the_table = axes[1].table(cellText=cell_text,
                      rowLabels=legendlabels,
                      rowColours=colors,
                      colLabels=['dj38', 'qa194', 'd198', 'lin318', 'pcb442', 'rat783', 'pr1002', 'pcb1173', 'd1291', 'pr2392'],
                      loc='bottom')
the_table.auto_set_font_size(False)
the_table.set_fontsize(14)
cell_text_ll = []
cell_text_ll.append(['%1.1f' % (x) for x in percents_ll[0]])
cell_text_ll.append(['%1.1f' % (x) for x in percents_ll[1]])
the_table = axes[0].table(cellText=cell_text_ll,
                      rowLabels=legendlabels,
                      rowColours=['#A60628', '#f95d7f'],
                      colLabels=['dj38', 'qa194', 'd198', 'lin318', 'pcb442', 'rat783', 'pr1002', 'pcb1173', 'd1291', 'pr2392'],
                      loc='bottom')
the_table.auto_set_font_size(False)
the_table.set_fontsize(14)
for x in range(2):
    # Create green Bars
    a = percents[x]
    axes[1].bar(pos, a, color=colors[x], bottom=bottoms, edgecolor='white', width=barWidth, label=legendlabels[x])
    bottoms = bottoms + percents[x]

width = 0.45
#axes[0].set_xticks((pos + (width / 2)) - 0.15)
#axes[1].set_xticks((pos + (width / 2)) - 0.15)
plt.xticks([])
axes[1].set_yticks(np.arange(0, 110, step=10))
axes[0].set_yticks(np.arange(0, 110, step=10))
#axes[0].set_xticklabels(['dj38', 'qa194', 'd198', 'lin318', 'pcb442', 'rat783', 'pr1002', 'pcb1173', 'd1291', 'pr2392'])
#axes[1].set_xticklabels(['dj38', 'qa194', 'd198', 'lin318', 'pcb442', 'rat783', 'pr1002', 'pcb1173', 'd1291', 'pr2392'])
axes[1].set_facecolor('#E6E6E6')
axes[0].set_facecolor('#E6E6E6')
# Put a legend below current axis
#axes[1].legend(loc='center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5)
#axes[0].legend(loc='center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5)
fig.tight_layout(pad=5.0)
# 1.05, -1.30
for ax in axes.flat:
    ax.label_outer()
for ax in fig.get_axes():
    ax.label_outer()
# Show graphic
plt.subplots_adjust(hspace=0.5, left=0.125)
plt.show()