'''
When run, this file reads through the responses in the response directory,
parses them, evaluates the answers it finds, and records the results in the
evaluation directory. The response directory is assumed to have a structure
matching what generate.py generates in the prompt directory, optionally with
subdirectories for multiple repeats (see ../test.py). If any model/problem set/
frame combination already has a corresponding file in the evaluation directory,
it will not be re-evaluated unless the file is deleted. The file overrides.txt
directly inside the response directory, if it exists, is read in at the
beginning of execution and can specify overrides to the automatic parser's
output for specific response files. At the end of execution, any files in which
the automatic parser did not detect a coherent answer are added to
overrides.txt for optional manual review.

This file does not run with any arguments.
'''

from os import path, listdir
import numpy as np

from metadata import *
from utils import get_clean_tokens, grouping_to_string, string_to_grouping,\
     string_to_preference_matrix, sorted_grouping, grouping_is_stable

def preprocess(lines):
  if len(lines) == 0:
    return []
  
  if lines[0] == '<think>':
    i = len(lines) - 1
    while True:
      if i <= 0:
        return []
      
      if lines[i] == '</think>':
        lines = lines[i + 1:]
        break
      
      i -= 1
  
  tokens = []
  
  for line in lines:
    line = line.strip()
    
    if line.startswith(r'\boxed{') and line.endswith('}'):
      line = line.removeprefix(r'\boxed{').removesuffix('}')
      revised_lines = [subline.strip() for subline in line.split(r'\\')]
    else:
      revised_lines = [line]
    
    for revised_line in revised_lines:
      line_tokens = get_clean_tokens(revised_line)
      if len(line_tokens) >= 1:
        tokens.append(line_tokens)
  
  return tokens

def _grouping_dict_to_tuples(n, k, grouping):
  group_list = [[] for j in range(n)]
  for index, group_num in grouping.items():
    group_list[group_num].append(index)
  return sorted_grouping(group_list)

def get_roommates_grouping(lines, n, k):
  nk = n * k
  name_indices = {}
  for i, v in enumerate(male_names_lower[:nk]):
    name_indices[v] = i
    name_indices[v[0]] = i
  
  # gpt-oss-120b can make these mistakes
  if 'charlie' in name_indices:
    name_indices['charllie'] = name_indices['charlie']
  if 'dave' in name_indices:
    name_indices['david'] = name_indices['dave']
  
  tokens = preprocess(lines)
  
  if len(tokens) == 0:
    return 'not_found'
  
  grouping = {}
  
  i = len(tokens) - 1
  while i >= 0:
    line_tokens = tokens[i]
    
    if len(line_tokens) == 1 and line_tokens[0] == 'impossible':
      return 'impossible'
    
    line_tokens = [t for t in line_tokens if t != 'and']
    
    group_num = len(grouping) // k
    group = set()
    group_candidates = []
    if len(line_tokens) >= k:
      group_candidates.append(line_tokens[:k])
      if len(line_tokens) > k:
        group_candidates.append(line_tokens[-k:])
    
    for group_candidate in group_candidates:
      for token in group_candidate:
        if token in name_indices and name_indices[token] not in grouping\
        and name_indices[token] not in group:
          group.add(name_indices[token])
        else:
          group = set()
          break
      else:
        break
    
    if len(group) == k:
      for index in group:
        grouping[index] = group_num
      if len(grouping) == nk:
        return _grouping_dict_to_tuples(n, k, grouping)
    else:
      grouping = {}
    
    i -= 1
  
  i = len(tokens) - 1
  while i >= 0:
    line_tokens = tokens[i]
    
    grouping = {}
    group_num = 0
    group = set()
    
    j = 0
    while j < len(line_tokens):
      token = line_tokens[j]
      
      if token == 'impossible':
        return 'impossible'
      
      if token != 'and':
        if token in name_indices and name_indices[token] not in grouping\
        and name_indices[token] not in group:
          group.add(name_indices[token])
          if len(group) == k:
            for index in group:
              grouping[index] = group_num
            group_num += 1
            group = set()
        else:
          group = set()
        
        if len(grouping) == nk:
          return _grouping_dict_to_tuples(n, k, grouping)
      
      j += 1
    
    i -= 1
  '''
  for line_tokens in tokens:
    print(line_tokens)
    print([name_indices[t] for t in line_tokens if t in name_indices])
  print()
  '''
  return 'not_found'

frame_grouping_funcs = [get_roommates_grouping]

overrides = {}

if path.exists(overrides_path):
  overrides_file = open(overrides_path, 'r')
  
  for line in overrides_file:
    line = line.strip()
    
    if len(line) == 0:
      continue
    
    model, ps_short_name, frame, repeat, filename, grouping = line.split()
    overrides[(model, ps_short_name, frame, repeat, filename)] = grouping
  
  overrides_file.close()

for model in models:
  for problem_set in problem_sets:
    ps_short_name = problem_set['short_name']
    n = problem_set['n']
    k = problem_set['k']
    
    ps_data_dir = path.join(data_dir, ps_short_name)
    mps_response_dir = path.join(response_dir, model, ps_short_name)
    
    if not path.exists(ps_data_dir) or not path.exists(mps_response_dir):
      continue
    
    eval_file_paths = []
    mps_frame_indices = []
    repeat_dirs = []
    
    for i in range(len(frames)):
      f_response_dir = path.join(mps_response_dir, frames[i])
      eval_file_paths.append(path.join(evaluation_dir,
        '%s_%s_%s.txt' % (model, ps_short_name, frames[i])))
      
      if not path.exists(f_response_dir) or path.exists(eval_file_paths[-1]):
        continue
      
      mps_frame_indices.append(i)
      f_repeat_dirs = []
      
      for filename in listdir(f_response_dir):
        file_path = path.join(f_response_dir, filename)
        if path.isdir(file_path):
          f_repeat_dirs.append(file_path)
      
      if len(f_repeat_dirs) == 0:
        f_repeat_dirs.append(f_response_dir)
      
      repeat_dirs.append(f_repeat_dirs)
    
    if len(mps_frame_indices) == 0:
      continue
    
    print('Evaluating %s %s' % (model, ps_short_name))
    
    filenames = []
    stable_counts = []
    answers = [[[] for j in range(len(repeat_dirs[m]))]
               for m in range(len(mps_frame_indices))]
    
    ### temp
    incorrect = {'overall': 0}
    fs = {'overall': 0}
    wg = {'overall': 0}
    fu = {'overall': 0}
    nf = {'overall': 0}
    total = {'overall': 0}
    ###
    
    for filename in listdir(ps_data_dir):
      filenames.append(filename)
      
      data_file_path = path.join(ps_data_dir, filename)
      data_file = open(data_file_path, 'r')
      
      lines = data_file.read().splitlines()
      
      data_file.close()
      
      prefs_str = '\n'.join(lines[:-1])
      prefs = string_to_preference_matrix(prefs_str)
      
      num_stable = int(lines[-1].split()[0])
      stable_counts.append(num_stable)
      
      for m in range(len(mps_frame_indices)):
        i = mps_frame_indices[m]
        ###
        if num_stable not in total:
          incorrect[num_stable] = 0
          fs[num_stable] = 0
          wg[num_stable] = 0
          fu[num_stable] = 0
          nf[num_stable] = 0
          total[num_stable] = 0
        total['overall'] += 1
        total[num_stable] += 1
        ###
        for j in range(len(repeat_dirs[m])):
          file_path = path.join(repeat_dirs[m][j], filename)
          
          if not path.exists(file_path):
            answers[m][j].append(None)
            continue
          
          repeat = path.basename(repeat_dirs[m][j])
          overrides_key = (model, ps_short_name, frames[i], repeat, filename)
          grouping = None
          
          if overrides_key in overrides:
            grouping = overrides[overrides_key]
            if grouping == 'not_found':
              del overrides[overrides_key]
              grouping = None
          
          if grouping is None:
            file = open(file_path, 'r', encoding='utf-8')
            lines = file.read().splitlines()
            file.close()
            
            grouping = frame_grouping_funcs[i](lines, n, k)
          
          if isinstance(grouping, str) and grouping.startswith('not_found'):
            overrides[overrides_key] = grouping
            if grouping == 'not_found--refuse':
              evaluation = 'refuse'
            else:
              evaluation = 'incorrect'
          elif grouping == 'impossible':
            if num_stable > 0:
              evaluation = 'incorrect'
            else:
              evaluation = 'correct'
          else: # grouping is an actual attempt at a valid grouping
            if isinstance(grouping, str):
              grouping = string_to_grouping(grouping)
            
            if num_stable > 0\
            and grouping_is_stable(prefs, grouping):
              evaluation = 'correct'
            else:
              evaluation = 'incorrect'
            
            grouping = grouping_to_string(grouping)
          
          answers[m][j].append((grouping, evaluation))
          
          ###
          if evaluation == 'incorrect':
            incorrect['overall'] += 1
            incorrect[num_stable] += 1
            
            error_dict = None
            if isinstance(grouping, str) and grouping.startswith('not_found'):
              error_dict = nf
            elif num_stable == 0:
              error_dict = fs
            elif grouping == 'impossible':
              error_dict = fu
            else:
              error_dict = wg
            error_dict['overall'] += 1
            error_dict[num_stable] += 1
          ###
    
    ###
    ic = incorrect['overall']
    t = total['overall']
    print('overall incorrect %d %d %.3f (fs %.3f wg %.3f fu %.3f nf %.3f)'\
          % (ic, t, ic/t, fs['overall']/t, wg['overall']/t, fu['overall']/t,
             nf['overall']/t))
    del total['overall']
    for num_stable in sorted(total):
      ic = incorrect[num_stable]
      t = total[num_stable]
      print('%d incorrect %d/%d %.3f (fs %.3f wg %.3f fu %.3f nf %.3f)'\
            % (num_stable, ic, t, ic/t, fs[num_stable]/t, wg[num_stable]/t,
               fu[num_stable]/t, nf[num_stable]/t))
    ###
    
    for m in range(len(mps_frame_indices)):
      i = mps_frame_indices[m]
      eval_file = open(eval_file_paths[i], 'w')
      for k in range(len(filenames)):
        for j in range(len(repeat_dirs[m])):
          answer = answers[m][j][k]
          
          if answer is None:
            continue
          
          print('%s %s %d %s %s'\
                % (filenames[k], path.basename(repeat_dirs[m][j]),
                   stable_counts[k], answer[0], answer[1]),
                file=eval_file)
      
      eval_file.close()

overrides_file = open(overrides_path, 'w')

for key, grouping in sorted(overrides.items()):
  model, ps_short_name, frame, repeat, filename = key
  print('%s %s %s %s %s %s' % (model, ps_short_name, frame, repeat, filename,
                               grouping), file=overrides_file)

overrides_file.close()
