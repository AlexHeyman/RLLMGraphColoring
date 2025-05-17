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

from metadata import *
from utils import get_clean_tokens, coloring_to_string, string_to_coloring,\
     coloring_is_valid

def tolerate_imprecision_by_default(model):
  if model == 'claude3.7s-think'\
  or model.startswith('o1') or model.startswith('o3'):
    return False
  return True

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

def get_math_coloring(lines, num_vertices, connected_vertices, num_colors,
                      model):
  vertices = {v: i for i, v in enumerate(str(i) for i in range(num_vertices))}
  tolerate = tolerate_imprecision_by_default(model)
  
  tokens = preprocess(lines)
  
  if len(tokens) == 0:
    return 'not_found'
  
  coloring = {}
  non_arbitrary_vertices = 0
  
  i = len(tokens) - 1
  
  if tolerate and 'impossible' in tokens[i]:
    return 'impossible'
  
  while i >= 0:
    if len(tokens[i]) == 1 and tokens[i][0] == 'impossible':
      return 'impossible'
    
    line_tokens = tokens[i]
    
    if line_tokens[0] == 'vertex':
      line_tokens = line_tokens[1:]
    
    if len(line_tokens) >= 2 and line_tokens[0] in vertices:
      vertex = vertices[line_tokens[0]]
      line_tokens = line_tokens[1:]
      
      if len(line_tokens) >= 2 and line_tokens[0] == 'is':
        line_tokens = line_tokens[1:]
      
      if len(line_tokens) >= 2 and line_tokens[0] == 'colored':
        line_tokens = line_tokens[1:]
      
      for j in range(num_colors):
        if math_colors_lower[j].startswith(line_tokens[0]):
          coloring[vertex] = j
          non_arbitrary_vertices += 1
          break
      else:
        if tolerate:
          coloring[vertex] = 0 # Arbitrary
        else:
          coloring = {}
          non_arbitrary_vertices = 0
      
      if len(coloring) == num_vertices:
        if non_arbitrary_vertices > 0:
          return coloring
        else:
          coloring = {}
          non_arbitrary_vertices = 0
    elif len(coloring) > 0:
      coloring_vertices = set(coloring)
      if len(connected_vertices - coloring_vertices) == 0\
      and non_arbitrary_vertices > 0:
        # Coloring includes all connected vertices
        for vertex in range(num_vertices):
          if vertex not in coloring:
            coloring[vertex] = 0 # Arbitrary
        
        return coloring
      else:
        coloring = {}
        non_arbitrary_vertices = 0
    
    i -= 1
  
  # If a partial coloring was found right at the beginning of the file,
  # check for inclusion of all connected vertices
  coloring_vertices = set(coloring)
  if len(connected_vertices - coloring_vertices) == 0:
    for vertex in range(num_vertices):
      if vertex not in coloring:
        coloring[vertex] = 0
    
    return coloring
  
  return 'not_found'

def get_cities_coloring(lines, num_vertices, connected_vertices, num_colors,
                        model):
  vertices = {v: i for i, v in enumerate(
    str(i) for i in range(1, num_vertices + 1))}
  vp_index = cities_colors_lower.index('vp')
  tolerate = tolerate_imprecision_by_default(model)
  
  tokens = preprocess(lines)
  
  if len(tokens) == 0:
    return 'not_found'
  
  coloring = {}
  non_arbitrary_vertices = 0
  
  i = len(tokens) - 1
  
  if tolerate and 'impossible' in tokens[i]:
    return 'impossible'
  
  while i >= 0:
    if len(tokens[i]) == 1 and tokens[i][0] == 'impossible':
      return 'impossible'
    
    line_tokens = tokens[i]
    
    if line_tokens[0] == 'city':
      line_tokens = line_tokens[1:]
    
    if len(line_tokens) >= 2 \
    and line_tokens[0].isdigit() and line_tokens[1] == 'city':
      line_tokens = line_tokens[2:]
    
    if len(line_tokens) >= 2 and line_tokens[0] in vertices:
      vertex = vertices[line_tokens[0]]
      for j in range(num_colors):
        if cities_colors_lower[j].startswith(line_tokens[1]):
          coloring[vertex] = j
          non_arbitrary_vertices += 1
          break
      else:
        if vp_index < num_colors and len(line_tokens) >= 3 \
        and line_tokens[1] == 'vice' and line_tokens[2] == 'president':
          coloring[vertex] = vp_index
          non_arbitrary_vertices += 1
        else:
          if tolerate:
            coloring[vertex] = 0 # Arbitrary
          else:
            coloring = {}
            non_arbitrary_vertices = 0
      
      if len(coloring) == num_vertices:
        if non_arbitrary_vertices > 0:
          return coloring
        else:
          coloring = {}
          non_arbitrary_vertices = 0
    elif len(coloring) > 0:
      coloring_vertices = set(coloring)
      if len(connected_vertices - coloring_vertices) == 0\
      and non_arbitrary_vertices > 0:
        # Coloring includes all connected vertices
        for vertex in range(num_vertices):
          if vertex not in coloring:
            coloring[vertex] = 0 # Arbitrary

        return coloring
      else:
        coloring = {}
        non_arbitrary_vertices = 0
    
    i -= 1
  
  # If a partial coloring was found right at the beginning of the file,
  # check for inclusion of all connected vertices
  coloring_vertices = set(coloring)
  if len(connected_vertices - coloring_vertices) == 0:
    for vertex in range(num_vertices):
      if vertex not in coloring:
        coloring[vertex] = 0
    
    return coloring
  
  return 'not_found'

def get_friends_coloring(lines, num_vertices, connected_vertices, num_colors,
                         model):
  vertices = {v: i for i, v in enumerate(friends_names_lower[:num_vertices])}
  tolerate = tolerate_imprecision_by_default(model)
  
  tokens = preprocess(lines)
  
  if len(tokens) == 0:
    return 'not_found'
  
  coloring = {}
  non_arbitrary_vertices = 0
  
  i = len(tokens) - 1
  
  if tolerate and 'impossible' in tokens[i]:
    return 'impossible'
  
  while i >= 0:
    if len(tokens[i]) == 1 and tokens[i][0] == 'impossible':
      return 'impossible'
    
    line_tokens = tokens[i]
    
    if line_tokens[0].isdigit():
      line_tokens = line_tokens[1:]
    
    if len(line_tokens) >= 2 and line_tokens[0] in vertices:
      vertex = vertices[line_tokens[0]]
      for j in range(num_colors):
        if friends_colors_lower[j].startswith(line_tokens[1]):
          coloring[vertex] = j
          non_arbitrary_vertices += 1
          break
      else:
        if tolerate:
          coloring[vertex] = 0 # Arbitrary
        else:
          coloring = {}
          non_arbitrary_vertices = 0
      
      if len(coloring) == num_vertices:
        if non_arbitrary_vertices > 0:
          return coloring
        else:
          coloring = {}
          non_arbitrary_vertices = 0
    elif len(coloring) > 0:
      coloring_vertices = set(coloring)
      if len(connected_vertices - coloring_vertices) == 0\
      and non_arbitrary_vertices > 0:
        # Coloring includes all connected vertices
        for vertex in range(num_vertices):
          if vertex not in coloring:
            coloring[vertex] = 0 # Arbitrary
        
        return coloring
      else:
        coloring = {}
        non_arbitrary_vertices = 0
    
    i -= 1
  
  # If a partial coloring was found right at the beginning of the file,
  # check for inclusion of all connected vertices
  coloring_vertices = set(coloring)
  if len(connected_vertices - coloring_vertices) == 0:
    for vertex in range(num_vertices):
      if vertex not in coloring:
        coloring[vertex] = 0
    
    return coloring
  
  return 'not_found'

frame_coloring_funcs = [get_math_coloring, get_math_coloring,
                        get_cities_coloring, get_friends_coloring]

overrides = {}

if path.exists(overrides_path):
  overrides_file = open(overrides_path, 'r')
  
  for line in overrides_file:
    line = line.strip()
    
    if len(line) == 0:
      continue
    
    model, ps_short_name, frame, repeat, filename, coloring = line.split()
    overrides[(model, ps_short_name, frame, repeat, filename)] = coloring
  
  overrides_file.close()

for model in models:
  for problem_set in problem_sets:
    ps_short_name = problem_set['short_name']
    num_vertices = problem_set['num_vertices']
    num_colors = problem_set['num_colors']
    
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
    possible = []
    answers = [[[] for j in range(len(repeat_dirs[m]))]
               for m in range(len(mps_frame_indices))]
    
    for filename in listdir(ps_data_dir):
      filenames.append(filename)
      
      data_file_path = path.join(ps_data_dir, filename)
      data_file = open(data_file_path, 'r')
      
      lines = data_file.read().splitlines()
      
      data_file.close()
      
      edges = [tuple(int(vertex) for vertex in edge.split(','))
               for edge in lines[0].split('|')]
      connected_vertices = set(vertex for edge in edges for vertex in edge)
      
      problem_is_possible = (lines[1] == 'True')
      possible.append(problem_is_possible)
      
      for m in range(len(mps_frame_indices)):
        i = mps_frame_indices[m]
        for j in range(len(repeat_dirs[m])):
          file_path = path.join(repeat_dirs[m][j], filename)
          
          if not path.exists(file_path):
            answers[m][j].append(None)
            continue
          
          repeat = path.basename(repeat_dirs[m][j])
          overrides_key = (model, ps_short_name, frames[i], repeat, filename)
          coloring = None
          
          if overrides_key in overrides:
            coloring = overrides[overrides_key]
            if coloring == 'not_found':
              del overrides[overrides_key]
              coloring = None
          
          if coloring is None:
            file = open(file_path, 'r', encoding='utf-8')
            lines = file.read().splitlines()
            file.close()
            
            coloring = frame_coloring_funcs[i](
              lines, num_vertices, connected_vertices, num_colors, model)
          
          if isinstance(coloring, str) and coloring.startswith('not_found'):
            overrides[overrides_key] = coloring
            if coloring == 'not_found--refuse':
              evaluation = 'refuse'
            else:
              evaluation = 'incorrect'
          elif coloring == 'impossible':
            if problem_is_possible:
              evaluation = 'incorrect'
            else:
              evaluation = 'correct'
          else: # coloring is an actual attempt at a valid coloring
            if isinstance(coloring, str):
              coloring = string_to_coloring(coloring)
            
            if problem_is_possible\
            and coloring_is_valid(num_vertices, edges, coloring):
              evaluation = 'correct'
            else:
              evaluation = 'incorrect'
            
            coloring = coloring_to_string(coloring)
          
          answers[m][j].append((coloring, evaluation))
    
    for m in range(len(mps_frame_indices)):
      i = mps_frame_indices[m]
      eval_file = open(eval_file_paths[i], 'w')
      for k in range(len(filenames)):
        for j in range(len(repeat_dirs[m])):
          answer = answers[m][j][k]
          
          if answer is None:
            continue
          
          print('%s %s %s %s %s'\
                % (filenames[k], path.basename(repeat_dirs[m][j]),
                   possible[k], answer[0], answer[1]),
                file=eval_file)
      
      eval_file.close()

overrides_file = open(overrides_path, 'w')

for key, coloring in sorted(overrides.items()):
  model, ps_short_name, frame, repeat, filename = key
  print('%s %s %s %s %s %s' % (model, ps_short_name, frame, repeat, filename,
                               coloring), file=overrides_file)

overrides_file.close()
