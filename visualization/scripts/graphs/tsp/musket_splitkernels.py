import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

my_data=pd.read_csv('../../../data_aggregation/TSP/HighLevel//Musket_1,3,5-12_splitkernels.csv', delimiter=',', header=None)


labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 35, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]
width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots(sharey=True)

many_years = my_data.iloc[0]
labels = [str((year)) for year in many_years]
ind = np.arange(len(labels))

legendlabels = ["" for x in range(10)]
#legendlabels = pd.DataFrame(np.zeros((1, 10)))
legendlabels[0] = 'Initialize Datastructures and Skeletons'
legendlabels[1] = 'Read Data and Copy to Device'
legendlabels[2] = 'Calculate Distance'
legendlabels[3] = 'Calculate Iroueltte'
legendlabels[4] = 'Route Kernel'
legendlabels[5] = 'Update Best Sequence'
legendlabels[6] = 'Update Pheromones'
legendlabels[7] = 'Minimum Kernel'
legendlabels[8] = 'Update Pheromones'

#legendlabels[9] = 'Update Pheromones'

counter = 0
dump_data = pd.DataFrame(np.zeros((0, 10)))
for x in range(1, 11):
    dump_data = my_data.iloc[[x]].astype(float).values.tolist()
    if counter != 0:
        cummulateddata = pd.DataFrame(np.zeros((1, 9)))
        dump_data2 = [0, 0, 0, 0, 0, 0, 0, 0]
        for y in range(1, counter+1):
            currentrow = my_data.iloc[[y]].astype(float).values.tolist()
            for z in range(0, 8):
                dump_data2[z] = dump_data2[z] + currentrow[0][z]
        ax.bar(labels, dump_data[0], width, bottom=dump_data2, label=legendlabels[counter])
    else:
        ax.bar(labels, dump_data[0], width, label=legendlabels[counter])
    counter = counter + 1

ax.set_ylabel('ms')
ax.set_title('Runtimes by kernel')
ax.legend()

plt.show()
