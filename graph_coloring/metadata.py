from os import path

from utils import compute_vertex_conns, color_greedy

data_dir = 'data'
prompt_dir = 'prompts'
response_dir = 'responses'
overrides_path = path.join(response_dir, 'overrides.txt')
evaluation_dir = 'evaluations'
summary_dir = 'summaries'

models = ['o1-mini',
          'o3-mini-low',
          'o3-mini-medium',
          'o3-mini-high',
          'deepseek-r1',
          'claude3.7s-think',
          'gemini2.5pp',
          'grok3mb-low',
          'grok3mb-high'
          ]

def cf_8v4c_hec_col(edges):
  num_vertices = 8
  num_colors = 4
  vertex_conns = compute_vertex_conns(num_vertices, edges)
  coloring = color_greedy(num_vertices, vertex_conns, num_colors)
  return (coloring is not None)

def cf_8v4c_adv(edges):
  num_vertices = 8
  num_colors = 4
  
  if (6, 7) in edges:
    return False
  
  vertex_conns = compute_vertex_conns(num_vertices, edges)
  coloring = color_greedy(num_vertices, vertex_conns, num_colors)
  
  if coloring is None:
    return False
  
  vertex_conns[6].append(7)
  vertex_conns[7].append(6)
  
  coloring = color_greedy(num_vertices, vertex_conns, num_colors)
  return (coloring is None)

# random_method is either 'efficient' (default) or 'legacy'
# condition_func only has an effect when random_method == 'efficient'
# search_for_fes is 'all' (default), 'false_uncolorable', or 'none'
problem_sets = [{'name': '4 vertices, 2 colors',
                 'short_name': '4v2c',
                 'num_vertices': 4,
                 'num_colors': 2
                 },
                {'name': '8 vertices, 4 colors',
                 'short_name': '8v4c',
                 'num_vertices': 8,
                 'num_colors': 4,
                 'max_samples_per_ec': 50,
                 'random_method': 'legacy',
                 'random_seed': 102
                 },
                {'name': '8v4c greedy score test',
                 'short_name': '8v4c_gs_test',
                 'num_vertices': 8,
                 'num_colors': 4,
                 'edge_counts': [19, 20],
                 'max_samples_per_ec': 1000,
                 'random_seed': 2,
                 'frames': ['math']
                 },
                {'name': '8v4c high edge count colorable',
                 'short_name': '8v4c_hec_col',
                 'num_vertices': 8,
                 'num_colors': 4,
                 'edge_counts': [20, 21, 22, 23],
                 'max_samples_per_ec': 100,
                 'condition_func': cf_8v4c_hec_col,
                 'random_seed': 3,
                 'frames': ['math', 'friends'],
                 'search_for_fes': 'false_uncolorable'
                 },
                {'name': '8v4c adversarial',
                 'short_name': '8v4c_adv',
                 'num_vertices': 8,
                 'num_colors': 4,
                 'edge_counts': [20, 21, 22, 23],
                 'max_samples_per_ec': 100,
                 'condition_func': cf_8v4c_adv,
                 'random_seed': 4,
                 'frames': ['math', 'friends'],
                 'search_for_fes': 'false_uncolorable'
                 },
                {'name': '12 vertices, 6 colors',
                 'short_name': '12v6c',
                 'num_vertices': 12,
                 'num_colors': 6,
                 'edge_counts': list(range(47, 56 + 1)),
                 'max_samples_per_ec': 50,
                 'random_seed': 6,
                 'frames': ['math', 'friends']
                 }
                ]

ps_by_short_name = {ps['short_name']: ps for ps in problem_sets}

frames = ['math', 'friends']
frame_indices = {v: i for i, v in enumerate(frames)}

math_colors = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Purple']
math_colors_lower = [s.lower() for s in math_colors]

friends_colors = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Purple']
friends_colors_lower = [s.lower() for s in friends_colors]
friends_names = ['Alice', 'Bob', 'Carol', 'Dave', 'Ethan', 'Fran', 'George',
                 'Heather', 'Irene', 'Jack', 'Kathy', 'Larry']
friends_names_lower = [s.lower() for s in friends_names]
