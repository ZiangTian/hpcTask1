import glob
import math
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean


# Read data
data = None
pathlist = glob.glob('NumPy Benchmarks - Line Count.csv') 
for f in pathlist:
	ndata = pd.read_csv(f)
	if data is None:
		data = ndata
	else:
		data = pd.concat([data, ndata])

print(data)


# make three datasets: one with total, one with diff, one with percetage changed, all in same order
data['Pythran Perc'] = (data['Pythran Diff']/data['NumPy Tot'])*100
data['Numba Perc'] = (data['Numba Diff']/data['NumPy Tot'])*100
data['CuPy Perc'] = (data['CuPy Diff']/data['NumPy Tot'])*100
data['DaCe Perc'] = (data['DaCe Diff']/data['NumPy Tot'])*100
data['NumPy Perc'] = data['NumPy Tot'] - data['NumPy Tot']


# color of the heatmap is percentage changed
colors = data[['Pythran Perc', 'Numba Perc', 'CuPy Perc', 'DaCe Perc', 'NumPy Perc']]
# rename the columns, drop the " Perc" for labelling
colors = colors.rename(columns={"Pythran Perc": "Pythran", "Numba Perc": "Numba", 'CuPy Perc' : "CuPy", 'DaCe Perc': 'DaCe', 'NumPy Perc': 'NumPy'})
# reorder the columns to match the other plot
colors = colors[['CuPy', 'DaCe', 'Numba', 'Pythran', 'NumPy']]

# number in the heatmap is change to NumPy (except for NumPy, where it is the total)
numbers = data[['Pythran Tot', 'DaCe Tot', 'Numba Tot', 'CuPy Tot', 'NumPy Tot']]
numbers = numbers.rename(columns={"Pythran Tot": "Pythran", "Numba Tot": "Numba", 'CuPy Tot' : "CuPy", 'DaCe Tot': 'DaCe', 'NumPy Tot': 'NumPy'})
for i in ['Pythran', 'DaCe', 'Numba', 'CuPy']:
	numbers[i] = numbers[i] - numbers['NumPy']
numbers = numbers[['CuPy', 'DaCe', 'Numba', 'Pythran', 'NumPy']]


plt.style.use('classic')
fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6, 12))


# plot benchmark heatmap
im = ax0.imshow(colors.to_numpy(), cmap='RdYlGn_r', interpolation='nearest', aspect="auto", vmin=0, vmax=100)

for i in range(len(data['Benchmark'])):
	for j in range(len(colors.columns)):
		l = numbers.to_numpy()[i,j]
		lo = l
		p = colors.to_numpy()[i,j]
		if not math.isnan(p):
			p = str(int(p))
		if j < len(colors.columns)-1:
			if math.isnan(l):
				text = ax0.text(j, i, "unsupported", ha="center", va="center", color="red", fontsize=7)
			elif l>=0:
				l = "+" + str(int(l))
			else:
				l = str(int(l)) #+ ", " + str(p) + "%"
			if not math.isnan(lo):
				text = ax0.text(j, i, l, ha="center", va="center", color="white", fontsize=10)
		else:
			if not math.isnan(lo):
				text = ax0.text(j, i, int(l), ha="center", va="center", color="white", fontweight='bold', fontsize=10)

# We want to show all ticks...
ticks = ax0.set_xticks(np.arange(len(colors.columns)))
ticks = ax0.set_yticks(np.arange(len(data['Benchmark'])))
# ... and label them with the respective list entries
ticks = ax0.set_xticklabels(colors.columns)
ticks = ax0.set_yticklabels(data['Benchmark'])

# Rotate the tick labels and set their alignment.
plt.setp(ax0.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.tight_layout()
plt.savefig("plot2.pdf", dpi=600)
plt.show()
