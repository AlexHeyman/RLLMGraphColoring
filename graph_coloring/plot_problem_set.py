'''
When run, this file generates a graph of the relationship between edge count
and fraction of problems that are colorable for a specified problem set, and
opens it in a new window. It also prints to the console the total number of
problems with each edge count, the number of colorable problems with each edge
count, and the number of colorable problems belonging to each greedy score
category (or for which a greedy score has not been computed).

This file accepts one argument, which is the short name of the problem set to
plot. If unspecified, it defaults to '4v2c'.
'''

import sys
from os import path, listdir
from math import ceil
import matplotlib.pyplot as plt

from metadata import *

ps_short_name = '4v2c'
if len(sys.argv) >= 2:
  ps_short_name = sys.argv[1]

graph_width = 7
graph_height = 4
border_size = 0.025

ps_data_dir = path.join(data_dir, ps_short_name)

edge_counts = []
ec_set = set()
possible = []
greedy_scores = []
gs_category_counts = {'g ≥ 0.9': 0, '0.5 ≤ g < 0.9': 0, 'g < 0.5': 0,
                      'uncomputed': 0}

for filename in listdir(ps_data_dir):
  data_file_path = path.join(ps_data_dir, filename)
  data_file = open(data_file_path, 'r')
  
  lines = data_file.read().splitlines()
  
  data_file.close()
  
  edges = [tuple(int(num) for num in edge.split(','))
           for edge in lines[0].split('|')]
  num_edges = len(edges)
  edge_counts.append(num_edges)
  ec_set.add(num_edges)
  
  is_possible = (lines[1] == 'True')
  possible.append(is_possible)
  
  greedy_score = None
  if is_possible:
    if len(lines) >= 3:
      greedy_score = float(lines[2])
      if greedy_score >= 0.9:
        gs_category_counts['g ≥ 0.9'] += 1
      elif greedy_score >= 0.5:
        gs_category_counts['0.5 ≤ g < 0.9'] += 1
      else:
        gs_category_counts['g < 0.5'] += 1
    else:
      gs_category_counts['uncomputed'] += 1
  else:
    greedy_score = -1
  
  greedy_scores.append(greedy_score)

ec_all = {ec: 0 for ec in ec_set}
ec_possible = {ec: 0 for ec in ec_set}

for problem_index in range(len(edge_counts)):
  num_edges = edge_counts[problem_index]
  is_possible = possible[problem_index]
  ec_all[num_edges] += 1
  if is_possible:
    ec_possible[num_edges] += 1

ec_list = sorted(ec for ec in ec_set)

print('Problems by edge count: %s (total: %d)'\
      % ({ec: ec_all[ec] for ec in ec_list}, sum(ec_all.values())))
print('Colorable problems by edge count: %s (total: %d)'\
      % ({ec: ec_possible[ec] for ec in ec_list}, sum(ec_possible.values())))
print('Greedy scores for colorable problems: %s' % gs_category_counts)

fig, ax = plt.subplots()
fig.set_size_inches(graph_width, graph_height)

legend_order = []

ec_ticks = list(ec_list)
ec_list_zeros = [0 for _ in ec_list]

if max(ec_possible[i] for i in ec_list) > 0:
  label = 'Colorable'
  legend_order.append(label)
  ax.fill_between(ec_list, ec_list_zeros,
                  [ec_possible[i] / ec_all[i] for i in ec_list],
                  color=str(6/8), label=label, zorder=-1)

handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles),
                              key=lambda t: legend_order.index(t[0])))
ax.legend(handles, labels, loc='lower left')

width = ec_list[-1] - ec_list[0]
left = ec_list[0] - border_size * width
right = ec_list[-1] + border_size * width
ax.set_xlim(left, right)
ax.set_ylim(0 - border_size, 1 + border_size)
ax.set_title(ps_short_name)
ax.set_xlabel('Number of edges')
ax.set_xticks(ec_ticks)
ax.set_ylabel('Fraction of problems')

fig.tight_layout()
fig.show()
