'''
When run, this file generates a histogram of the numbers of stable groupings
for each problem in a specified problem set, and opens it in a new window. It
also prints to the console the number of problems in the problem set with each
represented number of stable groupings.

This file accepts one argument, which is the short name of the problem set to
plot. If unspecified, it defaults to '2g3'.
'''

import sys
from os import path, listdir
import matplotlib.pyplot as plt

from metadata import *

ps_short_name = '2g3'
if len(sys.argv) >= 2:
  ps_short_name = sys.argv[1]

graph_width = 7
graph_height = 5

ps_data_dir = path.join(data_dir, ps_short_name)

stable_counts = []
sc_frequencies = {}

for filename in listdir(ps_data_dir):
  data_file_path = path.join(ps_data_dir, filename)
  data_file = open(data_file_path, 'r')
  lines = data_file.read().splitlines()
  data_file.close()
  
  num_stable = int(lines[-1].split()[0])
  stable_counts.append(num_stable)
  if num_stable in sc_frequencies:
    sc_frequencies[num_stable] += 1
  else:
    sc_frequencies[num_stable] = 1

unique_sc_list = sorted(sc_frequencies.keys())

for sc in unique_sc_list:
  print('%d stable groupings: %d problems' % (sc, sc_frequencies[sc]))

bins = [n - 0.5 for n in range(unique_sc_list[0], unique_sc_list[-1] + 1)]
bins.append(unique_sc_list[-1] + 0.5)

fig, ax = plt.subplots()
fig.set_size_inches(graph_width, graph_height)

ax.hist(stable_counts, bins=bins)

ax.set_title('%s: Numbers of stable groupings in problems' % ps_short_name)
ax.set_xlabel('Number of stable groupings')
ax.set_ylabel('Number of problems')

fig.tight_layout()
fig.show()
