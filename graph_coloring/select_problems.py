'''
When run, this file copies files from a specified "source" subdirectory of the
prompt directory to another specified "destination" subdirectory, selecting
which files to copy and which to leave behind based on a specified criterion.
The user can also specify which frame subdirectories to copy, and whether to
also copy the corresponding files between subdirectories of the data directory
with the same names.

This file does not run with any arguments; parameters are specified in a
section at the start of the code.
'''

from os import path, listdir, mkdir
from shutil import copyfile
import math
import itertools
import random

from metadata import *
from utils import 

### Specify parameters here

source_dir_name = '8v4c_gs_test'
dest_dir_name = '8v4c_gs_test_med'
copy_data = True
frames_to_copy = ['math']

def decide(filename, random_index, lines):
  if random_index >= 250:
    return False
  
  edges = string_to_edges(lines[0])
  num_edges = len(edges)
  is_possible = (lines[1] == 'True')
  
  if is_possible and len(lines) >= 3:
    gs = float(lines[2])
    return (gs < 0.9 and gs >= 0.5)
  
  return False

###

source_data_dir = path.join(data_dir, source_dir_name)
source_prompt_dir = path.join(prompt_dir, source_dir_name)
dest_data_dir = path.join(data_dir, dest_dir_name)
dest_prompt_dir = path.join(prompt_dir, dest_dir_name)

if copy_data and not path.exists(dest_data_dir):
  mkdir(dest_data_dir)

if len(frames_to_copy) > 0 and not path.exists(dest_prompt_dir):
  mkdir(dest_prompt_dir)

for frame in frames_to_copy:
  frame_dir = path.join(dest_prompt_dir, frame)
  if not path.exists(frame_dir):
    mkdir(frame_dir)

filenames = list(listdir(source_data_dir))
random.shuffle(filenames)

random_index = 0

for filename in filenames:
  data_file_path = path.join(source_data_dir, filename)
  data_file = open(data_file_path, 'r')
  
  lines = data_file.read().splitlines()
  
  data_file.close()
  
  if decide(filename, random_index, lines):
    random_index += 1
    
    if copy_data:
      source_path = path.join(source_data_dir, filename)
      dest_path = path.join(dest_data_dir, filename)
      copyfile(source_path, dest_path)
    
    for frame in frames_to_copy:
      source_path = path.join(source_prompt_dir, frame, filename)
      dest_path = path.join(dest_prompt_dir, frame, filename)
      copyfile(source_path, dest_path)
