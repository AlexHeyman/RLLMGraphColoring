'''
When run, this file reports whether a specified grouping is stable for a
specified problem, and which pairs of entities are switchable if it is unstable.

This file accepts three arguments. The first is the short name of the problem
set that contains the problem (e.g. "2g3"). The second is the name of the
problem's file within the problem set's data directory (e.g. "1.txt"). The third
is the grouping to assess, with entities using their single-character names and
groups separated by hyphens (e.g. "ABC-DEF").
'''

import sys
from os import path

from metadata import *
from utils import grouping_to_string, string_to_grouping,\
     string_to_preference_matrix, possible_intergroup_pairs, sorted_grouping,\
     group_value

assert(len(sys.argv) == 4)

ps_short_name = sys.argv[1]
filename = sys.argv[2]
grouping_str = sys.argv[3]

problem_set = ps_by_short_name[ps_short_name]
n = problem_set['n']
k = problem_set['k']
nk = n * k

grouping = string_to_grouping(grouping_str)
assert(sum(len(group) for group in grouping) == nk)
assert(set(range(nk)) == set(i for group in grouping for i in group))
grouping = sorted_grouping(grouping)

print('Assessing %s %s: grouping %s' % (ps_short_name, filename,
                                        grouping_to_string(grouping)))

ps_data_dir = path.join(data_dir, ps_short_name)
data_file_path = path.join(ps_data_dir, filename)
data_file = open(data_file_path, 'r')

lines = data_file.read().splitlines()

data_file.close()

prefs_str = '\n'.join(lines[:-1])
prefs = string_to_preference_matrix(prefs_str)

print()
print(prefs_str)
print()

num_switchable = 0

for i1, j1, i2, j2 in possible_intergroup_pairs(grouping):
  group1 = grouping[i1]
  e1 = group1[j1]
  others1 = group1[:j1] + group1[j1 + 1:]
  group2 = grouping[i2]
  e2 = group2[j2]
  others2 = group2[:j2] + group2[j2 + 1:]
  
  value1 = group_value(prefs, e1, others1)
  value1_alt = group_value(prefs, e1, others2)
  value2 = group_value(prefs, e2, others2)
  value2_alt = group_value(prefs, e2, others1)
  
  if (value1_alt > value1 and value2_alt >= value2)\
  or (value2_alt > value2 and value1_alt >= value1):
    num_switchable += 1
    c1 = index_chars[e1]
    c2 = index_chars[e2]
    print('%s and %s are switchable (%s %d -> %d, %s %d -> %d)'\
          % (c1, c2, c1, value1, value1_alt, c2, value2, value2_alt))

if num_switchable == 0:
  print('Grouping is stable')
else:
  print('Grouping is unstable')
