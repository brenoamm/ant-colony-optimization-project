import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'../../.'
my_data = pd.read_csv('../../../data_aggregation/TSP/HighLevel/Musket_route_kernel_average.csv', delimiter=',', header=None)
breno_data = pd.read_csv('../../../data_aggregation/TSP/LowLevel/Lowlevel_route_kernel_average.csv', delimiter=',', header=None)

#define size
#fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 10))

# X-Achsis
many_years = my_data.iloc[0]
years = [str(int(year)) for year in many_years][9:]
ind = np.arange(len(years))    # the x locations for the groups
width = 0.35         # the width of the bars

# Data from Musket
Musket1 = my_data.iloc[1].astype(float)[9:]
Musket2 = my_data.iloc[2].astype(float)[9:]
Musket4 = my_data.iloc[3].astype(float)[9:]
Musket8 = my_data.iloc[4].astype(float)[9:]

bars1 = [Musket1[9],Musket2[9],Musket4[9],Musket8[9]]

# Data Breno
Breno1 = breno_data.iloc[1].astype(float)[9:]
Breno2 = breno_data.iloc[2].astype(float)[9:]
Breno4 = breno_data.iloc[3].astype(float)[9:]
Breno8 = breno_data.iloc[4].astype(float)[9:]

bars2 = [Breno1[9],Breno2[9],Breno4[9],Breno8[9]]

fig, ax = plt.subplots()
ax.set_axisbelow(True)
ax.grid()

# width of the bars
barWidth = 0.3

# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(9, 3))

# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = '#A60628', edgecolor = 'black', capsize=7, label='Musket')
plt.bar(r2, bars2, width = barWidth, color = '#348ABD', edgecolor = 'black', capsize=7, label='Low-level')

# general layout
plt.xticks([r + barWidth/2 for r in range(len(bars1))], ['1024', '2048', '4096', '8192'])
plt.title('Route Kernel Runtime Comparison - pr2392')
plt.ylabel('seconds')
plt.xlabel('ants')
plt.legend()

plt.show()