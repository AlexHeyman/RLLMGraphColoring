'''
When run, this file reads through the files in the evaluation directory and
generates plots summarizing the results, putting them in the summary directory.

This file does not run with any arguments.
'''

from os import path, listdir, mkdir
from math import ceil
from scipy.stats import binomtest
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from metadata import *
from utils import string_to_edges

def get_stats(positive, total):
  prop = positive / total
  ci = binomtest(positive, total).proportion_ci(confidence_level=0.95)
  return prop, (prop - ci.low), (ci.high - prop)

plt.rcdefaults()
plt.rcParams['font.family'] = 'Times New Roman'

max_graph_width = 5.5 * 2

models_to_load = ['o1-mini', 'deepseek-r1', 'claude3.7s-think',
                  'o3-mini-low', 'o3-mini-medium', 'o3-mini-high',
                  'gemini2.5pp', 'grok3mb-low', 'grok3mb-high']
pses_to_load = ['4v2c', '8v4c', '8v4c_hec_col', '8v4c_adv', '12v6c',
                '8v4c_gs_test']
frames_to_load = ['math', 'friends']

models_with_false_edge_data = ['deepseek-r1', 'claude3.7s-think',
  'o3-mini-high', 'gemini2.5pp', 'grok3mb-low', 'grok3mb-high']
models_with_hidden_CoT = ['o1-mini', 'o3-mini-low', 'o3-mini-medium',
                          'o3-mini-high', 'gemini2.5pp']
models_with_exposed_CoT = ['deepseek-r1', 'claude3.7s-think',
                           'grok3mb-low', 'grok3mb-high']

model_proper_names = ['o1-mini', 'DeepSeek-R1', 'Claude 3.7 Sonnet (thinking)',
                      'o3-mini (low)', 'o3-mini (medium)', 'o3-mini (high)',
                      'Gemini 2.5 Pro Preview', 'Grok 3 Mini Beta (low)',
                      'Grok 3 Mini Beta (high)']
model_proper_names = {models_to_load[i]: model_proper_names[i]\
                      for i in range(len(models_to_load))}

model_short_names = ['o1-mini', 'DeepSeek-R1', 'Claude 3.7 S',
                     'o3-mini (low)', 'o3-mini (med)', 'o3-mini (high)',
                     'Gemini 2.5 PP', 'Grok 3 MB (low)', 'Grok 3 MB (high)']
model_short_names = {models_to_load[i]: model_short_names[i]\
                     for i in range(len(models_to_load))}

frame_proper_names = ['Math', 'Friends']
frame_proper_names = {frames_to_load[i]: frame_proper_names[i]\
                      for i in range(len(frames_to_load))}

filenames = {}
problem_edges = {}
problem_possible = {}
problem_greedy_scores = {}
evals = {}
features = {}

for ps_short_name in pses_to_load:
  filenames[ps_short_name] = []
  problem_edges[ps_short_name] = {}
  problem_possible[ps_short_name] = {}
  problem_greedy_scores[ps_short_name] = {}
  evals[ps_short_name] = {}
  features[ps_short_name] = {}
  
  # print('Getting data for %s' % ps_short_name)
  
  ps_data_dir = path.join(data_dir, ps_short_name)
  
  for filename in listdir(ps_data_dir):
    filenames[ps_short_name].append(filename)
    
    data_file_path = path.join(ps_data_dir, filename)
    data_file = open(data_file_path, 'r')
    lines = data_file.read().splitlines()
    data_file.close()
    
    edges = [tuple(int(num) for num in edge.split(','))
             for edge in lines[0].split('|')]
    problem_edges[ps_short_name][filename] = edges
    
    is_possible = (lines[1] == 'True')
    problem_possible[ps_short_name][filename] = is_possible
    
    if is_possible and len(lines) >= 3 and len(lines[2].strip()) > 0:
      greedy_score = float(lines[2])
      problem_greedy_scores[ps_short_name][filename] = greedy_score
  
  for model in models_to_load:
    mps_eval_paths = {}
    
    for frame in frames_to_load:
      mpsf_eval_path = path.join(evaluation_dir,
        '%s_%s_%s.txt' % (model, ps_short_name, frame))
      if path.exists(mpsf_eval_path):
        mps_eval_paths[frame] = mpsf_eval_path
    
    if len(mps_eval_paths) == 0:
      continue
    
    evals[ps_short_name][model] = {}
    
    for frame, mpsf_eval_path in mps_eval_paths.items():
      mpsf_evals = {}
      evals[ps_short_name][model][frame] = mpsf_evals
      
      # print('Getting evaluations for %s %s %s' % (ps_short_name, model, frame))
      
      mpsf_eval_file = open(mpsf_eval_path, 'r')
      
      for line in mpsf_eval_file:
        line = line.strip()
        
        if len(line) == 0:
          continue
        
        filename, repeat, is_possible, coloring, evaluation = line.split()
        
        mpsf_evals[(filename, repeat)] = (coloring, evaluation)
      
      mpsf_eval_file.close()
      
      mpsf_features_path = path.join(evaluation_dir, 'response_features',
        '%s_%s_%s.txt' % (model, ps_short_name, frame))
      if path.exists(mpsf_features_path):
        if model not in features[ps_short_name]:
          features[ps_short_name][model] = {}
        
        mpsf_features = {}
        features[ps_short_name][model][frame] = mpsf_features
        
        # print('Getting response features for %s %s %s'\
        #       % (ps_short_name, model, frame))
        
        mpsf_features_file = open(mpsf_features_path, 'r')
        
        for line in mpsf_features_file:
          line = line.strip()
          
          if len(line) == 0:
            continue
          
          filename, repeat, length, mentions_graph, manual_fes_search,\
            true_edges, false_edges, would_be_correct = line.split()
          
          mpsf_features[(filename, repeat)] = (length, mentions_graph,
            manual_fes_search, string_to_edges(true_edges),
            string_to_edges(false_edges), would_be_correct)
        
        mpsf_features_file.close()

# Make graph of 8v4c error types for each model/frame/colorability

models_to_plot = models_to_load
ps_to_plot = '8v4c'
frames_to_plot = ['math', 'friends']

type_names = ['False uncolorable', 'Wrong coloring', 'False colorable']
type_colors = ['red', (1, 0.25, 1), 'dodgerblue']

columns = 3
rows = ceil(len(models_to_plot) / columns)
fig, axs = plt.subplots(rows, columns)
fig.set_size_inches(max_graph_width, 4 * 2)

col = 0
row = 0
legend = {}

for model in models_to_plot:
  if rows == 1:
    ax = axs[col]
  else:
    ax = axs[row][col]
  
  bar_x = [[] for _ in range(3)]
  bar_h = [[] for _ in range(3)]
  err_x = []
  err_y = []
  err_lower = []
  err_upper = []
  xticks = []
  xticklabels = []
  
  x = 0
  
  for frame in frames_to_plot:
    numerators = [0 for _ in range(7)]
    denominators = [0 for _ in range(7)]
    
    for key, val in evals[ps_to_plot][model][frame].items():
      filename, repeat = key
      coloring, evaluation = val
      
      if problem_possible[ps_to_plot][filename]:
        denominators[0] += 1
        denominators[1] += 1
        denominators[2] += 1
        denominators[3] += 1
        if evaluation == 'incorrect':
          numerators[0] += 1
          if coloring.startswith('not_found'):
            numerators[3] += 1
          elif coloring == 'impossible':
            numerators[1] += 1
          else:
            numerators[2] += 1
      else:
        denominators[4] += 1
        denominators[5] += 1
        denominators[6] += 1
        if evaluation == 'incorrect':
          numerators[4] += 1
          if coloring.startswith('not_found'):
            numerators[6] += 1
          else:
            numerators[5] += 1
    
    assert(numerators[3] == 0)
    assert(numerators[6] == 0)
    
    rates = [numerators[i] / denominators[i] for i in range(7)]
    
    bar_x[0].append(x - 0.5)
    bar_h[0].append(rates[1])
    bar_x[1].append(x - 0.5)
    bar_h[1].append(rates[2])
    bar_x[2].append(x + 0.5)
    bar_h[2].append(rates[4])
    
    for i, r in enumerate([0, 4]):
      prop, lower, upper = get_stats(numerators[r], denominators[r])
      err_x.append(x - 0.5 + i)
      err_y.append(prop)
      err_lower.append(lower)
      err_upper.append(upper)
    
    xticks.append(x)
    xticklabels.append(frame_proper_names[frame])
    x += 3
  
  ax.bar(bar_x[0], bar_h[0], width=0.8, align='center', color=type_colors[0],
       edgecolor='k', linewidth=1, label=type_names[0])
  ax.bar(bar_x[1], bar_h[1], bottom=bar_h[0], width=0.8, align='center',
         color=type_colors[1], edgecolor='k', linewidth=1, label=type_names[1])
  ax.bar(bar_x[2], bar_h[2], width=0.8, align='center', color=type_colors[2],
       edgecolor='k', linewidth=1, label=type_names[2])
  ax.errorbar(err_x, err_y, [err_lower, err_upper],
              capsize=3, color='k', linestyle='none')
  
  ax.set_ylim(bottom=0)
  ax.set_ylabel('Error rate')
  ax.set_title(model_proper_names[model])
  ax.set_xticks(ticks=xticks, labels=xticklabels)
  
  handles, labels = ax.get_legend_handles_labels()
  for i in range(len(labels)):
    if labels[i] not in legend:
      legend[labels[i]] = handles[i]
  
  col += 1
  if col >= columns:
    row += 1
    col = 0

fig.legend(handles=legend.values(), labels=legend.keys(),
           loc='upper center', ncols=3)
fig.tight_layout()
fig.subplots_adjust(top=0.92)
save_path = path.join(summary_dir, '8v4c_error_types')
plt.savefig(save_path + '.png')
plt.savefig(save_path + '.pdf')
plt.close(fig)

# Make graph of 8v4c edge hallucinations by frame and edge count for models that
# expose CoT

models_to_plot = models_with_exposed_CoT
ps_to_plot = '8v4c'
frames_to_plot = ['math', 'friends']

model_colors = ['b', 'g', 'darkorange', 'r']

columns = 2
rows = ceil(len(frames_to_plot) / columns)
fig, axs = plt.subplots(rows, columns)
fig.set_size_inches(max_graph_width, 2 * 2)

col = 0
row = 0
legend = {}

for frame in frames_to_plot:
  if rows == 1:
    ax = axs[col]
  else:
    ax = axs[row][col]
  
  all_edge_counts = set()
  
  for model_i in range(len(models_to_plot)):
    model = models_to_plot[model_i]
    numerators = {}
    denominators = {}
    
    for key, val in features[ps_to_plot][model][frame].items():
      filename, repeat = key
      length, mentions_graph, manual_fes_search, true_edges, false_edges,\
        would_be_correct = val
      
      edge_count = len(problem_edges[ps_to_plot][filename])
      
      if false_edges is not None:
        if edge_count not in denominators:
          all_edge_counts.add(edge_count)
          numerators[edge_count] = 0
          denominators[edge_count] = 0
        
        denominators[edge_count] += 1
        if len(false_edges) > 0:
          numerators[edge_count] += 1
    
    xs = sorted(edge_count for edge_count in denominators)
    ys = []
    err_lower = []
    err_upper = []
    
    for x in xs:
      prop, lower, upper = get_stats(numerators[x], denominators[x])
      ys.append(prop)
      err_lower.append(lower)
      err_upper.append(upper)
    
    ax.errorbar(xs, ys, [err_lower, err_upper], capsize=3, marker='.',
                color=model_colors[model_i], label=model_proper_names[model])
  
  ax.set_ylim(0, 0.43)
  ax.set_ylabel('Rate of false edge inclusion in responses')
  ax.set_title(frame_proper_names[frame])
  ax.set_xticks(sorted(all_edge_counts))
  
  handles, labels = ax.get_legend_handles_labels()
  for i in range(len(labels)):
    if labels[i] not in legend:
      legend[labels[i]] = handles[i]
  
  col += 1
  if col >= columns:
    row += 1
    col = 0

fig.legend(handles=legend.values(), labels=legend.keys(),
           loc='upper center', ncols=4)
fig.tight_layout()
fig.subplots_adjust(top=0.84)
save_path = path.join(summary_dir, '8v4c_hallucinations')
plt.savefig(save_path + '.png')
plt.savefig(save_path + '.pdf')
plt.close(fig)

# Make graph of 8v4c false-uncolorable attributability for models with false
# edge data

models_to_plot = models_with_false_edge_data
ps_to_plot = '8v4c'
frames_to_plot = ['math', 'friends']

# frame_colors = ['deepskyblue', 'springgreen']
attr_names = ['Includes false edges', '+ correct under false edges']
attr_colors = ['orange', 'yellow']

'''
fig, axs = plt.subplots(2, 1)
fig.set_size_inches(max_graph_width, 3 * 2)
ax1, ax2 = axs
'''
fig, ax2 = plt.subplots()
fig.set_size_inches(max_graph_width, 1.75 * 2)
'''
ax1_bar_x = [[] for _ in range(len(frames_to_plot))]
ax1_bar_h = [[] for _ in range(len(frames_to_plot))]
ax1_err_x = []
ax1_err_y = []
ax1_err_lower = []
ax1_err_upper = []
ax1_xticks = []
ax1_xticklabels = []
'''
ax2_bar_x = []
ax2_bar_h1 = []
ax2_bar_h2 = []
ax2_bar_h1h2 = []
ax2_err_x = []
ax2_err_y = []
ax2_err_lower = []
ax2_err_upper = []
ax2_xticks_top = []
ax2_xticklabels_top = []
ax2_xticks_bottom = []
ax2_xticklabels_bottom = []

x = 0

for model in models_to_plot:
  initial_x = x
  CoT_hidden = (model in models_with_hidden_CoT)
  ax2_xs_unplotted = []
  
  for frame_i in range(len(frames_to_plot)):
    frame = frames_to_plot[frame_i]
    
    numerators = [0 for _ in range(4)]
    denominators = [0 for _ in range(4)]
    
    for key, val in features[ps_to_plot][model][frame].items():
      length, mentions_graph, manual_fes_search, true_edges, false_edges,\
        would_be_correct = val
      
      if false_edges is not None:
        denominators[0] += 1
        if len(false_edges) > 0:
          numerators[0] += 1
      
      coloring, evaluation = evals[ps_to_plot][model][frame][key]
      if coloring == 'impossible' and evaluation == 'incorrect':
        denominators[1] += 1
        denominators[2] += 1
        denominators[3] += 1
        if false_edges is not None and len(false_edges) > 0:
          numerators[1] += 1
          if would_be_correct == 'True':
            numerators[2] += 1
          else:
            numerators[3] += 1
    
    rates = [None for _ in range(4)]
    for i in range(4):
      if denominators[i] > 0:
        rates[i] = numerators[i] / denominators[i]
    '''
    if not CoT_hidden and denominators[0] > 0:
      ax1_bar_x[frame_i].append(x)
      ax1_bar_h[frame_i].append(rates[0])
      prop, lower, upper = get_stats(numerators[0], denominators[0])
      ax1_err_x.append(x)
      ax1_err_y.append(prop)
      ax1_err_lower.append(lower)
      ax1_err_upper.append(upper)
    '''
    if denominators[1] > 0:
      ax2_bar_x.append(x)
      ax2_bar_h1.append(rates[2])
      ax2_bar_h2.append(rates[3])
      ax2_bar_h1h2.append(rates[1])
      prop, lower, upper = get_stats(numerators[1], denominators[1])
      ax2_err_x.append(x)
      ax2_err_y.append(prop)
      ax2_err_lower.append(lower)
      ax2_err_upper.append(upper)
    else:
      ax2_xs_unplotted.append(x)
    
    ax2_xticks_bottom.append(x)
    ax2_xticklabels_bottom.append(frame_proper_names[frame])
    
    x += 1
  
  xtick = ((x - 1) + initial_x) / 2
  xticklabel = model_short_names[model]
  '''
  ax1_xticks.append(xtick)
  ax1_xticklabels.append(xticklabel)
  
  if CoT_hidden:
    ax1.text(xtick, 0.35/20, 'chain of thought hidden', color='k',
             fontsize=12, rotation='vertical', horizontalalignment='center',
             verticalalignment='bottom')
  '''
  ax2_xticks_top.append(xtick)
  ax2_xticklabels_top.append(xticklabel)
  
  for unplotted_x in ax2_xs_unplotted:
    ax2.text(unplotted_x, 1/20, 'denominator 0', color='k', fontsize=12,
             rotation='vertical', horizontalalignment='center',
             verticalalignment='bottom')
  
  x += 1
'''
for frame_i in range(len(frames_to_plot)):
  ax1.bar(ax1_bar_x[frame_i], ax1_bar_h[frame_i], width=0.8, align='center',
          color=frame_colors[frame_i], edgecolor='k', linewidth=1,
          label=frame_proper_names[frames_to_plot[frame_i]])

ax1.errorbar(ax1_err_x, ax1_err_y, [ax1_err_lower, ax1_err_upper],
             capsize=3, color='k', linestyle='none')
'''
ax2.bar(ax2_bar_x, ax2_bar_h2, bottom=ax2_bar_h1, width=0.8, align='center',
        color=attr_colors[0])
ax2.bar(ax2_bar_x, ax2_bar_h1, width=0.8, align='center', color=attr_colors[1])
ax2.bar(ax2_bar_x, ax2_bar_h1h2, width=0.8, align='center', color='none',
        edgecolor='k', linewidth=1)
ax2.errorbar(ax2_err_x, ax2_err_y, [ax2_err_lower, ax2_err_upper],
             capsize=3, color='k', linestyle='none')

ax2_legend_handles = [Patch(facecolor=attr_colors[0], edgecolor='k',
                            label=attr_names[0]),
                      Patch(facecolor=attr_colors[1], edgecolor='k',
                            label=attr_names[1])]
'''
ax1.legend(loc='upper center', ncols=2)
ax1.set_ylim(0, 0.35)
ax1.set_ylabel('Rate of false edge inclusion in responses')
ax1.set_xticks(ticks=ax1_xticks, labels=ax1_xticklabels)
'''
ax2.legend(handles=ax2_legend_handles, loc='upper center', ncols=2)
ax2.set_ylim(0, 1)
ax2.set_ylabel('Fraction of false-uncolorable responses')
ax2.set_xticks(ticks=ax2_xticks_bottom, labels=ax2_xticklabels_bottom)

ax3 = ax2.twiny()
ax3.set_xlim(ax2.get_xlim())
ax3.set_xticks(ticks=ax2_xticks_top, labels=ax2_xticklabels_top)

fig.tight_layout()
save_path = path.join(summary_dir, '8v4c_attributability')
plt.savefig(save_path + '.png')
plt.savefig(save_path + '.pdf')
plt.close(fig)

# Make graph of 8v4c_hec_col and 8v4c_adv errors and attributability for each
# model and frame

models_to_plot = models_to_load
pses_to_plot = ['8v4c', '8v4c_hec_col', '8v4c_adv']
frames_to_plot = ['math', 'friends']

ps_names = ['8v4c', '8v4c high-edge colorable', '8v4c adversarial']
attr_names = ['No coherent answer', 'Wrong coloring', 'False uncolorable',
              '+ includes false edges', '+ correct under false edges',
              'False uncolorable (no false edge data)']
attr_colors = ['gray', (1, 0.25, 1), 'red', 'orange', 'yellow', 'lightgray']
num_attr = len(attr_names)

columns = 3
rows = ceil(len(models_to_plot) / columns)
fig, axs = plt.subplots(rows, columns)
fig.set_size_inches(max_graph_width, 4 * 2)

col = 0
row = 0

for model in models_to_plot:
  if rows == 1:
    ax = axs[col]
  else:
    ax = axs[row][col]
  
  model_has_false_edge_data = (model in models_with_false_edge_data)
  
  bar_x = [[] for _ in range(num_attr)]
  all_bar_x = []
  bar_h = [[] for _ in range(num_attr)]
  bar_bottoms = [[] for _ in range(num_attr)]
  bar_tops = []
  err_lower = []
  err_upper = []
  
  xticks = []
  xticklabels = []
  
  x = 0
  
  for frame in frames_to_plot:
    initial_x = x
    
    for ps_short_name in pses_to_plot:
      all_bar_x.append(x)
      
      numerators = [0 for _ in range(num_attr)]
      denominator = 0
      
      for key, val in features[ps_short_name][model][frame].items():
        filename, repeat = key
        length, mentions_graph, manual_fes_search, true_edges, false_edges,\
          would_be_correct = val
        coloring, evaluation = evals[ps_short_name][model][frame][key]
        
        if not problem_possible[ps_short_name][filename]:
          continue
        
        denominator += 1
        
        if evaluation == 'incorrect':
          if coloring.startswith('not_found'):
            numerators[5] += 1
          elif coloring == 'impossible':
            if model_has_false_edge_data:
              if false_edges is not None and len(false_edges) > 0:
                if would_be_correct == 'True':
                  numerators[1] += 1
                else: # has false edges, but not correct under them
                  numerators[2] += 1
              else: # doesn't have false edges, but still false uncolorable
                numerators[3] += 1
            else: # false uncolorable with unknown false edge status
              numerators[0] += 1
          else: # wrong coloring rather than false uncolorable
            numerators[4] += 1
      
      partial_sum = 0
      
      for i in range(num_attr):
        if numerators[i] > 0:
          bar_x[i].append(x)
          bar_h[i].append(numerators[i] / denominator)
          bar_bottoms[i].append(partial_sum / denominator)
        partial_sum += numerators[i]
      
      prop, lower, upper = get_stats(partial_sum, denominator)
      bar_tops.append(prop)
      err_lower.append(lower)
      err_upper.append(upper)
      
      x += 1
    
    xticks.append((x - 1 + initial_x) / 2)
    xticklabels.append(frame_proper_names[frame])
    
    x += 1
  
  for i in range(num_attr):
    if len(bar_x[i]) > 0:
      ax.bar(bar_x[i], bar_h[i], bottom=bar_bottoms[i],
             width=0.8, align='center', color=attr_colors[num_attr - 1 - i])
  
  ax.bar(all_bar_x, bar_tops, width=0.8, align='center', color='none',
        edgecolor='k', linewidth=1)
  ax.errorbar(all_bar_x, bar_tops, [err_lower, err_upper],
              capsize=3, color='k', linestyle='none')
  
  ax.set_ylim(bottom=0)
  ax.set_ylabel('Error rate')
  ax.set_title(model_proper_names[model])
  ax.set_xticks(ticks=xticks, labels=xticklabels)
  
  col += 1
  if col >= columns:
    row += 1
    col = 0

legend_handles = [Patch(facecolor=attr_colors[i], edgecolor='k',
                        label=attr_names[i]) for i in range(num_attr)]

fig.legend(handles=legend_handles, loc='upper center', ncols=3)
fig.supxlabel('Each bar triplet: 8v4c colorable → high-edge-count colorable → uncolorable with highest-index edge added')
fig.tight_layout()
fig.subplots_adjust(top=0.89)
save_path = path.join(summary_dir, '8v4c_progression')
plt.savefig(save_path + '.png')
plt.savefig(save_path + '.pdf')
plt.close(fig)

# Make graph of 12v6c errors and attributability for o3-mini-high and
# grok3mb-high, and hallucinations for the latter, for both frames

models_to_plot = ['o3-mini-high', 'grok3mb-high']
ps_to_plot = '12v6c'
frames_to_plot = ['math', 'friends']

# frame_colors = ['deepskyblue', 'springgreen']
frame_colors = ['b', 'g']
attr_names = ['No coherent answer', 'False colorable', 'Wrong coloring',
              'False uncolorable', '+ includes false edges',
              '+ correct under false edges']
attr_colors = ['gray', 'dodgerblue', (1, 0.25, 1), 'red', 'orange', 'yellow']
num_attr = len(attr_names)

cols = 3
rows = 1
fig, axs = plt.subplots(rows, cols, width_ratios=[1, 1.5, 1])
fig.set_size_inches(max_graph_width, 2 * 2)

col = 0
row = 0

for model in models_to_plot:
  # Make subgraph showing edge hallucination rates by edge count for this model
  # if it exposes its chain of thought
  
  if model in models_with_exposed_CoT:
    if rows == 1:
      ax = axs[col]
    else:
      ax = axs[row][col]
    '''
    bar_x = [[] for _ in range(len(frames_to_plot))]
    bar_h = [[] for _ in range(len(frames_to_plot))]
    err_x = []
    err_y = []
    err_lower = []
    err_upper = []
    xticks = []
    xticklabels = []
    
    x = 0
    
    for frame_i in range(len(frames_to_plot)):
      frame = frames_to_plot[frame_i]
      
      numerator = 0
      denominator = 0
      
      for key, val in features[ps_to_plot][model][frame].items():
        length, mentions_graph, manual_fes_search, true_edges, false_edges,\
          would_be_correct = val
        if false_edges is not None:
          denominator += 1
          if len(false_edges) > 0:
            numerator += 1
      
      bar_x[frame_i].append(x)
      bar_h[frame_i].append(numerator / denominator)
      prop, lower, upper = get_stats(numerator, denominator)
      err_x.append(x)
      err_y.append(prop)
      err_lower.append(lower)
      err_upper.append(upper)
      
      xticks.append(x)
      xticklabels.append(frame_proper_names[frame])
      
      x += 1.5
    
    for frame_i in range(len(frames_to_plot)):
      ax.bar(bar_x[frame_i], bar_h[frame_i], width=1, align='center',
             color=frame_colors[frame_i], edgecolor='k', linewidth=1)
    
    ax.errorbar(err_x, err_y, [err_lower, err_upper],
                capsize=3, color='k', linestyle='none')
    
    ax.set_xlim(-1, x - 1.5 + 1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Rate of false edge inclusion in responses')
    ax.set_xticks(ticks=xticks, labels=xticklabels)
    ax.set_title('%s – false edges' % model_proper_names[model])
    '''
    all_edge_counts = set()
    
    for frame_i in range(len(frames_to_plot)):
      frame = frames_to_plot[frame_i]
      numerators = {}
      denominators = {}
      
      for key, val in features[ps_to_plot][model][frame].items():
        filename, repeat = key
        length, mentions_graph, manual_fes_search, true_edges, false_edges,\
          would_be_correct = val
        
        edge_count = len(problem_edges[ps_to_plot][filename])
        
        if false_edges is not None:
          if edge_count not in denominators:
            all_edge_counts.add(edge_count)
            numerators[edge_count] = 0
            denominators[edge_count] = 0
          
          denominators[edge_count] += 1
          if len(false_edges) > 0:
            numerators[edge_count] += 1
      
      xs = sorted(edge_count for edge_count in denominators)
      ys = []
      err_lower = []
      err_upper = []
      
      for x in xs:
        prop, lower, upper = get_stats(numerators[x], denominators[x])
        ys.append(prop)
        err_lower.append(lower)
        err_upper.append(upper)
      
      ax.errorbar(xs, ys, [err_lower, err_upper], capsize=3, marker='.',
                  color=frame_colors[frame_i], label=frame_proper_names[frame])
    
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Rate of false edge inclusion in responses')
    ax.set_title('%s – false edges' % model_proper_names[model])
    ax.set_xticks(sorted(all_edge_counts))
    
    col += 1
    if col >= columns:
      row += 1
      col = 0
  
  # Make subgraph showing errors + attributability for this model
  
  if rows == 1:
    ax = axs[col]
  else:
    ax = axs[row][col]
  
  bar_x = [[] for _ in range(num_attr)]
  all_bar_x = []
  bar_h = [[] for _ in range(num_attr)]
  bar_bottoms = [[] for _ in range(num_attr)]
  bar_tops = []
  err_lower = []
  err_upper = []
  
  xticks = []
  xticklabels = []
  
  x = 0
  
  for frame in frames_to_plot:
    all_bar_x.append(x)
    
    numerators = [0 for _ in range(num_attr)]
    denominator = 0
    
    for key, val in features[ps_to_plot][model][frame].items():
      filename, repeat = key
      length, mentions_graph, manual_fes_search, true_edges, false_edges,\
        would_be_correct = val
      coloring, evaluation = evals[ps_to_plot][model][frame][key]
      
      denominator += 1
      
      if evaluation == 'incorrect':
        if coloring.startswith('not_found'):
          numerators[5] += 1
        elif not problem_possible[ps_to_plot][filename]:
          numerators[4] += 1
        elif coloring == 'impossible':
          if false_edges is not None and len(false_edges) > 0:
            if would_be_correct == 'True':
              numerators[0] += 1
            else: # has false edges, but not correct under them
              numerators[1] += 1
          else: # doesn't have false edges, but still false uncolorable
            numerators[2] += 1
        else: # wrong coloring rather than false uncolorable
          numerators[3] += 1
    
    partial_sum = 0
    
    for i in range(num_attr):
      if numerators[i] > 0:
        bar_x[i].append(x)
        bar_h[i].append(numerators[i] / denominator)
        bar_bottoms[i].append(partial_sum / denominator)
      partial_sum += numerators[i]
    
    prop, lower, upper = get_stats(partial_sum, denominator)
    bar_tops.append(prop)
    err_lower.append(lower)
    err_upper.append(upper)
    
    xticks.append(x)
    xticklabels.append(frame_proper_names[frame])
    
    x += 1.5
  
  for i in range(num_attr):
    if len(bar_x[i]) > 0:
      ax.bar(bar_x[i], bar_h[i], bottom=bar_bottoms[i],
             width=1, align='center', color=attr_colors[num_attr - 1 - i])
  
  ax.bar(all_bar_x, bar_tops, width=1, align='center', color='none',
        edgecolor='k', linewidth=1)
  ax.errorbar(all_bar_x, bar_tops, [err_lower, err_upper],
              capsize=3, color='k', linestyle='none')
  
  ax.set_xlim(-1, x - 1.5 + 1)
  ax.set_ylim(bottom=0)
  ax.set_ylabel('Error rate')
  ax.set_title('%s – error types' % model_proper_names[model])
  ax.set_xticks(ticks=xticks, labels=xticklabels)
  
  col += 1
  if col >= columns:
    row += 1
    col = 0

legend_handles = [Patch(facecolor=attr_colors[i], edgecolor='k',
                        label=attr_names[i]) for i in range(num_attr)]

fig.legend(handles=legend_handles, loc='upper center', ncols=3)
fig.tight_layout()
fig.subplots_adjust(top=0.78)
save_path = path.join(summary_dir, '12v6c')
plt.savefig(save_path + '.png')
plt.savefig(save_path + '.pdf')
plt.close(fig)

# Make graph of greedy score tests for o1-mini and DeepSeek-R1

models_to_plot = ['o1-mini', 'deepseek-r1']
ps_to_plot = '8v4c_gs_test'
frame_to_plot = 'math'
edge_counts_to_plot = [19, 20]
num_ec_to_plot = len(edge_counts_to_plot)

type_names = ['g ≥ 0.9', '0.5 ≤ g < 0.9', 'g < 0.5']
type_colors = ['royalblue', 'turquoise', 'limegreen']
num_types = len(type_names)

fig, ax = plt.subplots()
fig.set_size_inches(max_graph_width, 2 * 2)

bar_x = [[] for _ in range(num_types)]
bar_h = [[] for _ in range(num_types)]
err_x = []
err_y = []
err_lower = []
err_upper = []
xticks = []
xticklabels = []

x = 0

for model in models_to_plot:
  numerators = [[0 for _ in range(num_types)] for _2 in range(num_ec_to_plot)]
  denominators = [[0 for _ in range(num_types)] for _2 in range(num_ec_to_plot)]
  
  for key, val in evals[ps_to_plot][model][frame_to_plot].items():
    filename, repeat = key
    coloring, evaluation = val
    
    if problem_possible[ps_to_plot][filename]:
      num_edges = len(problem_edges[ps_to_plot][filename])
      edges_index = edge_counts_to_plot.index(num_edges)
      
      greedy_score = problem_greedy_scores[ps_to_plot][filename]
      if greedy_score >= 0.9:
        problem_type = 0
      elif greedy_score >= 0.5:
        problem_type = 1
      else:
        problem_type = 2
      
      denominators[edges_index][problem_type] += 1
      if evaluation == 'incorrect':
        numerators[edges_index][problem_type] += 1
  
  rates = [[numerators[i][j] / denominators[i][j] for j in range(num_types)]
           for i in range(num_ec_to_plot)]
  
  for i in range(num_ec_to_plot):
    initial_x = x
    for j in range(num_types):
      bar_x[j].append(x)
      bar_h[j].append(rates[i][j])
      
      prop, lower, upper = get_stats(numerators[i][j], denominators[i][j])
      err_x.append(x)
      err_y.append(prop)
      err_lower.append(lower)
      err_upper.append(upper)
      
      x += 1
    
    xticks.append((x - 1 + initial_x) / 2)
    xticklabels.append('%s (%d edges)'\
                       % (model_proper_names[model], edge_counts_to_plot[i]))
    x += 1

for i in range(num_types):
  ax.bar(bar_x[i], bar_h[i], width=0.8, align='center', color=type_colors[i],
       edgecolor='k', linewidth=1, label=type_names[i])

ax.errorbar(err_x, err_y, [err_lower, err_upper],
            capsize=3, color='k', linestyle='none')

ax.legend(loc='upper center', ncols=3)
ax.set_ylim(bottom=0)
ax.set_ylabel('Error rate')
ax.set_xticks(ticks=xticks, labels=xticklabels)

fig.tight_layout()
save_path = path.join(summary_dir, 'greedy_score_test')
plt.savefig(save_path + '.png')
plt.savefig(save_path + '.pdf')
plt.close(fig)

# Generate text file with statistics useful for tables

stats_file_path = path.join(summary_dir, 'stats.txt')
stats_file = open(stats_file_path, 'w')

for model in models_to_load:
  model_has_false_edge_data = (model in models_with_false_edge_data)
  model_has_exposed_CoT = (model in models_with_exposed_CoT)
  
  print('---', file=stats_file)
  print('Model: %s' % model_proper_names[model], file=stats_file)
  print('---', file=stats_file)
  print('Exposes CoT: %s' % model_has_exposed_CoT, file=stats_file)
  print('Has false edge data: %s' % model_has_false_edge_data, file=stats_file)
  print('', file=stats_file)
  
  for ps_short_name in evals:
    if model not in evals[ps_short_name]:
      continue
    
    problem_set = ps_by_short_name[ps_short_name]
    
    if 'search_for_fes' in problem_set:
      search_for_fes = problem_set['search_for_fes']
    else:
      search_for_fes = 'all'
    
    print('- Problem set: %s -' % problem_set['name'], file=stats_file)
    print('', file=stats_file)
    
    for frame in evals[ps_short_name][model]:
      print('Frame: %s' % frame_proper_names[frame], file=stats_file)
      
      mpsf_features = None
      if model in features[ps_short_name]\
      and frame in features[ps_short_name][model]:
        mpsf_features = features[ps_short_name][model][frame]
      
      eval_filenames = set()
      eval_possible_filenames = set()
      eval_total = 0
      es_total = 0
      es_hallucinated = 0
      eval_incorrect = 0
      eval_fc = 0
      eval_wc = 0
      eval_fu = 0
      eval_fu_hal = 0
      eval_fu_hal_wouldbe = 0
      false_edges_total = 0
      false_edge_freqs = {}
      
      for key, val in evals[ps_short_name][model][frame].items():
        filename, repeat = key
        coloring, evaluation = val
        
        eval_filenames.add(filename)
        if problem_possible[ps_short_name][filename]:
          eval_possible_filenames.add(filename)
        
        eval_total += 1
        if evaluation == 'incorrect':
          eval_incorrect += 1
          if coloring == 'impossible':
            eval_fu += 1 # false uncolorable
          elif problem_possible[ps_short_name][filename]:
            eval_wc += 1 # wrong coloring
          else:
            eval_fc += 1 # false colorable
        
        if mpsf_features is not None and key in mpsf_features:
          length, mentions_graph, manual_fes_search, true_edges, false_edges,\
            would_be_correct = mpsf_features[key]
          if false_edges is not None:
            es_total += 1
            if len(false_edges) > 0:
              es_hallucinated += 1
              
              for edge in false_edges:
                false_edges_total += 1
                if edge in false_edge_freqs:
                  false_edge_freqs[edge] += 1
                else:
                  false_edge_freqs[edge] = 1
              
              if evaluation == 'incorrect' and coloring == 'impossible':
                eval_fu_hal += 1
                if would_be_correct == 'True':
                  eval_fu_hal_wouldbe += 1
      
      if not model_has_exposed_CoT or search_for_fes != 'all':
        es_hallucinated = 'Unknown'
      
      if not model_has_false_edge_data or search_for_fes == 'none':
        eval_fu_hal = 'Unknown'
        eval_fu_hal_wouldbe = 'Unknown'
      
      print('Problems: %d (%d colorable)' % (len(eval_filenames),
        len(eval_possible_filenames)), file=stats_file)
      print('Trials: %s' % eval_total, file=stats_file)
      print('With false edges: %s' % es_hallucinated, file=stats_file)
      print('Incorrect: %s' % eval_incorrect, file=stats_file)
      print('False colorable: %s' % eval_fc, file=stats_file)
      print('Wrong coloring: %s' % eval_wc, file=stats_file)
      print('False uncolorable: %s' % eval_fu, file=stats_file)
      print('+ including false edges: %s' % eval_fu_hal, file=stats_file)
      print('+ correct under false edges: %s' % eval_fu_hal_wouldbe,
            file=stats_file)
      
      if false_edges_total > 0:
        print('False edge frequencies (top 10 at most):', file=stats_file)
        descending = sorted(false_edge_freqs,
                            key=lambda edge: -false_edge_freqs[edge])
        for i in range(min(len(descending), 10)):
          edge = descending[i]
          freq = false_edge_freqs[edge]
          print('%s: %d/%d (%.3f)' % (edge, freq, false_edges_total,
            freq / false_edges_total), file=stats_file)
      
      print('', file=stats_file)

stats_file.close()
