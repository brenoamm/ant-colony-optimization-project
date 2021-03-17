import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
# was my_data
runtimes = pd.read_csv('../../../data_aggregation/TSP/HighLevel/Musket_1,3,5-12_average.csv', delimiter=',', header=None)
runtimes_lowlevel = pd.read_csv('../../../data_aggregation/TSP/LowLevel/LowLevel_1,3,5-12_average.csv', delimiter=',', header=None)

# bmh -> forschig
# Solarize_Light2 angenehm anzusehen uebergaenge semi deutlich
# ggplot2 bessere unterscheidung farben semi schoen
mpl.style.use('bmh')
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_result_kernel_split_musket_averagekernelsmeans = [20, 35, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]
width = 0.35       # the width of the bars: can also be len(x) sequence

many_years = runtimes.iloc[0]
labels = [str(int(year)) for year in many_years]
xvalue = [(int(year)) for year in many_years][:-1]
yvalue_1024 = runtimes.iloc[[1]].astype(float).values[0][:-1]
yvalue_2048 = runtimes.iloc[[2]].astype(float).values[0][:-1]
yvalue_4096 = runtimes.iloc[[3]].astype(float).values[0][:-1]
yvalue_8192 = runtimes.iloc[[4]].astype(float).values[0][:-1]

yvalue_1024_ll = runtimes_lowlevel.iloc[[1]].astype(float).values[0][:-1]
yvalue_2048_ll = runtimes_lowlevel.iloc[[2]].astype(float).values[0][:-1]
yvalue_4096_ll = runtimes_lowlevel.iloc[[3]].astype(float).values[0][:-1]
yvalue_8192_ll = runtimes_lowlevel.iloc[[4]].astype(float).values[0][:-1]

ind = np.arange(len(labels))
# print yvalue_1024_ll

legendlabels = ["" for x in range(4)]
legendlabels[0] = '1024'
legendlabels[1] = '2048'
legendlabels[2] = '4096'
legendlabels[3] = '8192'
df = pd.DataFrame({'x': range(1, 11), 'y1': np.random.randn(10), 'y2': np.random.randn(10) + range(1, 11),
                   'y3': np.random.randn(10) + range(11, 21)})
#define size
plt.figure(figsize=(9, 3))

# multiple line plot
plt.plot(xvalue, yvalue_1024, marker='o', label = "1024", color='#348ABD')
plt.plot(xvalue, yvalue_2048, marker='o', label = "2048", color='#A60628')
plt.plot(xvalue, yvalue_4096, marker='o', label = "4096", color='#7A68A6')
plt.plot(xvalue, yvalue_8192, marker='o', label = "8192", color='#467821')

# different dash styles
dashes = [(None, None), [2, 2], [7, 4, 3, 4]]
         
line1,= plt.plot(xvalue, yvalue_1024_ll, linestyle='dashed', marker='o', label = "1024 ll", color='#348ABD')
line1.set_dashes(dashes[1])
line2,= plt.plot(xvalue, yvalue_2048_ll, linestyle='dashed', marker='o', label = "2048 ll", color='#A60628')
line2.set_dashes(dashes[1])
line3,= plt.plot(xvalue, yvalue_4096_ll, linestyle='dashed', marker='o', label = "4096 ll", color='#7A68A6')
line3.set_dashes(dashes[1])
line4,= plt.plot(xvalue, yvalue_8192_ll, linestyle='dashed', marker='o', label = "8192 ll", color='#467821')
line4.set_dashes(dashes[1])

plt.legend()
plt.xticks(xvalue)
#plt.plot( 'x', 'y4', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
#handles, labels = ax[1].get_legend_handles_labels()
#print handles
#fig.legend(handles, labels, loc='center right', prop={'size': 7})

plt.show()
