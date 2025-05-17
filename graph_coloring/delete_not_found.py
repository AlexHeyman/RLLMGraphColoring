'''
When run, this file examines responses with a specified model/problem set/frame
combination and deletes response files in which no coherent answer to the
problem was found. This file assumes that evaluate.py has already been run on
the relevant set of responses and the corresponding evaluation file exists in
the evaluation directory.

This file accepts three arguments, which are (in order) the model, problem set,
and frame to examine. If unspecified, they default to 'claude3.7s-think',
'8v4c', and 'math' respectively.
'''

import sys
from os import path, listdir, remove

from metadata import *

model = 'claude3.7s-think'
ps_short_name = '8v4c'
frame = 'math'
if len(sys.argv) >= 2:
  model = sys.argv[1]
  if len(sys.argv) >= 3:
    ps_short_name = sys.argv[2]
    if len(sys.argv) >= 4:
      frame = sys.argv[3]

not_found = []

eval_file_path = path.join(evaluation_dir,
                           '%s_%s_%s.txt' % (model, ps_short_name, frame))
eval_file = open(eval_file_path, 'r')

for line in eval_file:
  line = line.strip()
  
  if len(line) == 0:
    continue
  
  filename, repeat, is_possible, coloring, evaluation = line.split()
  
  if coloring == 'not_found':
    not_found.append((filename, repeat))

eval_file.close()

response_subdir = path.join(response_dir, model, ps_short_name, frame)

for filename, repeat in not_found:
  file_path = path.join(response_subdir, repeat, filename)
  
  if not path.exists(file_path) and repeat == path.basename(response_subdir):
    file_path = path.join(response_subdir, filename)
  
  if path.exists(file_path):
    print('Removing %s %s' % (repeat, filename))
    remove(file_path)
