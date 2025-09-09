'''
When run, this file generates sets of stable matching problems based on the
configuration in metadata.py, and stores basic data about the problems in the
data directory and prompts based on the problems in the prompt directory. If
any problem set already has data in the data directory, it will not be
re-generated unless the subdirectory with its data is deleted.

This file does not run with any arguments.
'''

from os import path, mkdir
import math
import itertools
import random
import numpy as np

from metadata import *
from utils import possible_groupings, possible_intergroup_pairs,\
     grouping_to_string, preference_matrix_to_string, get_stable_groupings

roommates_template = '''Suppose there are %d university students - %s - who must be grouped into %d dorm rooms holding %d students each. Suppose each student has rated how much they would want each other student as a roommate on a scale of 1 to 5, with 5 as the highest. The ratings can be found in the following matrix, with each row showing one student's ratings for all of the others:

%s

Suppose a student's satisfaction with their assigned room is equal to the sum of their ratings for their %d roommates. A grouping of the students into rooms is called unstable if there exists a pair of students in different rooms who could switch rooms to increase both of their satisfactions, or increase the satisfaction of one while the other's satisfaction stays the same. Do there exist any possible groupings that are stable (i.e. not unstable)? If there do not, write "Impossible" as the final line of your response. If there do, the final lines of your response should give an example of one in a format like the following:

%s
'''

for problem_set in problem_sets:
  ps_name = problem_set['name']
  ps_short_name = problem_set['short_name']
  n = problem_set['n']
  k = problem_set['k']
  nk = n * k
  num_problems = problem_set['num_problems']
  
  if 'frames' in problem_set:
    ps_frames = set(problem_set['frames'])
  else:
    ps_frames = set(frames)
  
  # Make data directory; if it already exists, skip generation
  
  ps_data_dir = path.join(data_dir, ps_short_name)
  
  if path.exists(ps_data_dir):
    continue
  
  print('Generating problem set: %s' % ps_name)
  mkdir(ps_data_dir)
  
  # Make prompt directory
  ps_prompt_dir = path.join(prompt_dir, ps_short_name)
  if not path.exists(ps_prompt_dir):
    mkdir(ps_prompt_dir)
  
  # Make subdirectories for frames
  for frame in ps_frames:
    frame_dir = path.join(ps_prompt_dir, frame)
    if not path.exists(frame_dir):
      mkdir(frame_dir)
  
  if 'random_seed' in problem_set:
    random.seed(problem_set['random_seed'])
    np.random.seed(problem_set['random_seed'])
  
  problem_number = 0
  while problem_number < num_problems:
    problem_number += 1
    
    # Generate preference matrix
    prefs = np.random.randint(1, 5 + 1, (nk, nk), dtype=np.uint8)
    for i in range(nk):
      prefs[i, i] = 0
    
    # Search exhaustively for stable groupings
    stable_groupings = get_stable_groupings(n, k, prefs)
    
    # Make data file
    data_file_path = path.join(ps_data_dir, '%d.txt' % problem_number)
    data_file = open(data_file_path, 'w')
    
    print(preference_matrix_to_string(prefs, index_chars), file=data_file)
    
    stable_groupings_str = str(len(stable_groupings))
    if len(stable_groupings) > 0:
      strs_to_print = []
      max_to_print = 10
      
      for i in range(min(len(stable_groupings), max_to_print)):
        strs_to_print.append(grouping_to_string(stable_groupings[i]))
      
      if len(stable_groupings) > max_to_print:
        strs_to_print.append('more')
      
      stable_groupings_str += ' (' + ', '.join(strs_to_print) + ')'
    
    print(stable_groupings_str, file=data_file)
    
    data_file.close()
    
    if 'roommates' in ps_frames:
      # Make prompt file for roommates frame
      frame_file_path = path.join(ps_prompt_dir, 'roommates',
                                  '%d.txt' % problem_number)
      frame_file = open(frame_file_path, 'w')
      
      names = male_names[:nk]
      
      names_str = ', '.join(names[:-1]) + ', and ' + names[-1]
      prefs_str = preference_matrix_to_string(prefs, names)
      example_str = '\n'.join('-'.join(names[k*i:k*(i+1)]) for i in range(n))
      
      prompt = roommates_template % (nk, names_str, n, k, prefs_str, n,
                                     example_str)
      frame_file.write(prompt)
      
      frame_file.close()
  
  print('Total: %d problems' % num_problems)
  print()
