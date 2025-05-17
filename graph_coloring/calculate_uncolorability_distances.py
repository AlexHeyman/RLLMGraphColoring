'''
When run, this file iterates through a specified problem set and, for each
problem, calculates and reports the minimum number of edges that must be added
to the graph before it becomes uncolorable. The calculated value maxes out at 3,
since the algorithm is worst-case exponential in the maximum value and the
distinction between non-small values is unlikely to be useful for understanding
errors induced by edge hallucination.

This file accepts one argument, which is the short name of the problem set to
examine. If not specified, it defaults to 4v2c.
'''

import sys
from os import path, listdir
import itertools

from metadata import *
from utils import compute_vertex_conns, color_greedy

ps_short_name = '4v2c'
if len(sys.argv) >= 2:
  ps_short_name = sys.argv[1]

def calculate_uncolorability_distance(num_vertices, edges_set, num_colors,
    edges_complement=None, recursion_depth=0, max_depth=2):
  
  if edges_complement is None:
    possible_edges = set(itertools.combinations(range(num_vertices), 2))
    edges_complement = possible_edges - edges_set
  
  vertex_conns = compute_vertex_conns(num_vertices, edges_set)
  coloring = color_greedy(num_vertices, vertex_conns, num_colors)
  
  if coloring is None:
    return recursion_depth
  elif recursion_depth >= max_depth:
    return max_depth + 1
  
  distances = []
  for edge in edges_complement:
    new_edges = set(edges_set)
    new_edges.add(edge)
    new_edges_comp = set(edges_complement)
    new_edges_comp.remove(edge)
    distances.append(calculate_uncolorability_distance(num_vertices,
      new_edges, num_colors, new_edges_comp, recursion_depth + 1, max_depth))
  
  return min(distances)

problem_set = ps_by_short_name[ps_short_name]
num_vertices = problem_set['num_vertices']
num_colors = problem_set['num_colors']

ps_data_dir = path.join(data_dir, ps_short_name)

for filename in listdir(ps_data_dir):
  data_file_path = path.join(ps_data_dir, filename)
  data_file = open(data_file_path, 'r')
  
  lines = data_file.read().splitlines()
  
  data_file.close()
  
  edges = [tuple(int(num) for num in edge.split(','))
           for edge in lines[0].split('|')]
  edges_set = set(edges)
  num_edges = len(edges)
  
  distance = calculate_uncolorability_distance(
    num_vertices, edges_set, num_colors)
  
  print('%s: %d edges, distance %d' % (filename, num_edges, distance))
