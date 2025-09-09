'''
When run, this file reads through the files in the evaluation directory and
generates plots summarizing the results, putting them in the summary directory.

This file does not run with any arguments.
'''

from os import path, listdir, mkdir
from math import ceil
import numpy as np
from scipy.stats import binomtest, ttest_1samp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from metadata import *
from utils import string_to_preference_matrix, string_to_rating_set

def get_stats(positive, total):
  prop = positive / total
  ci = binomtest(positive, total).proportion_ci(confidence_level=0.95)
  return prop, (prop - ci.low), (ci.high - prop)

plt.rcdefaults()
plt.rcParams['font.family'] = 'Times New Roman'

max_graph_width = 5.5 * 2

models_to_load = ['deepseek-r1', 'grok3mb-low', 'grok3mb-high',
                  'gpt-oss-120b-medium', 'gpt-oss-120b-high']
pses_to_load = ['2g3', '3g3']
frames_to_load = ['roommates']

model_proper_names = ['DeepSeek-R1', 'Grok 3 Mini Beta (low)',
                      'Grok 3 Mini Beta (high)', 'gpt-oss-120b (medium)',
                      'gpt-oss-120b (high)']
model_proper_names = {models_to_load[i]: model_proper_names[i]\
                      for i in range(len(models_to_load))}

model_short_names = ['DeepSeek-R1', 'Grok 3 MB (low)', 'Grok 3 MB (high)',
                     'gpt-oss-120b (medium)', 'gpt-oss-120b (high)']
model_short_names = {models_to_load[i]: model_short_names[i]\
                     for i in range(len(models_to_load))}

frame_proper_names = ['Roommates']
frame_proper_names = {frames_to_load[i]: frame_proper_names[i]\
                      for i in range(len(frames_to_load))}

filenames = {}
problem_prefs = {}
problem_stable_counts = {}
evals = {}
features = {}

for ps_short_name in pses_to_load:
  filenames[ps_short_name] = []
  problem_prefs[ps_short_name] = {}
  problem_stable_counts[ps_short_name] = {}
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
    
    prefs_str = '\n'.join(lines[:-1])
    prefs = string_to_preference_matrix(prefs_str)
    problem_prefs[ps_short_name][filename] = prefs
    
    num_stable = int(lines[-1].split()[0])
    problem_stable_counts[ps_short_name][filename] = num_stable
  
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
        
        filename, repeat, num_stable, grouping, evaluation = line.split()
        
        mpsf_evals[(filename, repeat)] = (grouping, evaluation)
      
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
          
          filename, repeat, length, manual_frs_search, true_ratings,\
            false_ratings, would_be_correct = line.split()
          
          mpsf_features[(filename, repeat)] = (length, manual_frs_search,
            string_to_rating_set(true_ratings),
            string_to_rating_set(false_ratings), would_be_correct)
        
        mpsf_features_file.close()

# Make graph of error types for each model and problem set

models_to_plot = models_to_load
pses_to_plot = pses_to_load
frame_to_plot = 'roommates'

type_names = ['False unsolvable', 'Wrong grouping', 'False solvable',
              'No coherent answer']
type_colors = ['red', (1, 0.25, 1), 'dodgerblue', 'gray']

columns = 5
rows = ceil(len(models_to_plot) * len(pses_to_plot) / columns)
fig, axs = plt.subplots(rows, columns)
fig.set_size_inches(max_graph_width, 3 * 2)

col = 0
row = 0
legend = {}

for ps in pses_to_plot:
  for model in models_to_plot:
    if rows == 1:
      ax = axs[col]
    else:
      ax = axs[row][col]
    
    bar_x = [[] for _ in range(4)]
    bar_h = [[] for _ in range(4)]
    err_x = []
    err_y = []
    err_lower = []
    err_upper = []
    
    numerators = [0 for _ in range(7)]
    denominators = [0 for _ in range(7)]
    
    for key, val in evals[ps][model][frame_to_plot].items():
      filename, repeat = key
      grouping, evaluation = val
      
      if problem_stable_counts[ps][filename] > 0:
        denominators[0] += 1
        denominators[1] += 1
        denominators[2] += 1
        denominators[3] += 1
        if evaluation == 'incorrect':
          numerators[0] += 1
          if grouping.startswith('not_found'):
            numerators[3] += 1
          elif grouping == 'impossible':
            numerators[1] += 1
          else:
            numerators[2] += 1
      else:
        denominators[4] += 1
        denominators[5] += 1
        denominators[6] += 1
        if evaluation == 'incorrect':
          numerators[4] += 1
          if grouping.startswith('not_found'):
            numerators[6] += 1
          else:
            numerators[5] += 1
    
    rates = [numerators[i] / denominators[i] for i in range(7)]
    
    bar_x[0].append(-0.5)
    bar_h[0].append(rates[1])
    bar_x[1].append(-0.5)
    bar_h[1].append(rates[2])
    bar_x[2].append(0.5)
    bar_h[2].append(rates[5])
    bar_x[3].extend([-0.5, 0.5])
    bar_h[3].extend([rates[3], rates[6]])
    
    for i, r in enumerate([0, 4]):
      prop, lower, upper = get_stats(numerators[r], denominators[r])
      err_x.append(-0.5 + i)
      err_y.append(prop)
      err_lower.append(lower)
      err_upper.append(upper)
    
    ax.bar(bar_x[0], bar_h[0], width=0.8, align='center', color=type_colors[0],
      edgecolor='k', linewidth=1, label=type_names[0])
    ax.bar(bar_x[1], bar_h[1], bottom=bar_h[0], width=0.8, align='center',
      color=type_colors[1], edgecolor='k', linewidth=1, label=type_names[1])
    ax.bar(bar_x[2], bar_h[2], width=0.8, align='center', color=type_colors[2],
      edgecolor='k', linewidth=1, label=type_names[2])
    ax.bar(bar_x[3], bar_h[3], bottom=[bar_h[0][0] + bar_h[1][0], bar_h[2][0]],
      width=0.8, align='center', color=type_colors[3], edgecolor='k',
      linewidth=1, label=type_names[3])
    ax.errorbar(err_x, err_y, [err_lower, err_upper],
                capsize=3, color='k', linestyle='none')
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(bottom=0)
    if col == 0:
      ax.set_ylabel('Error rate')
    ax.set_title('%s – %s' % (model_short_names[model], ps))
    ax.set_xticks(ticks=[], labels=[])
    
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
fig.subplots_adjust(top=0.89)
save_path = path.join(summary_dir, 'error_types')
plt.savefig(save_path + '.png')
plt.savefig(save_path + '.pdf')
plt.close(fig)

# Make graph of hallucination-related rates for each model and problem set

models_to_plot = models_to_load
pses_to_plot = pses_to_load
frame_to_plot = 'roommates'

prop_names = ['Incorrect', 'Has false ratings', 'Has false ratings | incorrect',
              'Would be correct | incorrect & has false ratings',
              'Transposed rating | false rating']
prop_colors = ['deeppink', 'green', 'orange', 'yellow', 'blueviolet']

columns = 1
rows = len(pses_to_plot)
fig, axs = plt.subplots(rows, columns)
fig.set_size_inches(max_graph_width, 3 * 2)

row = 0
legend = {}

for ps_short_name in pses_to_plot:
  ax = axs[row]
  
  bar_x = [[] for _ in range(len(prop_names))]
  bar_h = [[] for _ in range(len(prop_names))]
  err_x = []
  err_y = []
  err_lower = []
  err_upper = []
  xticks = []
  xticklabels = []
  
  x = 0
  
  for model in models_to_plot:
    numerators = [0 for _ in range(len(prop_names))]
    denominators = [0 for _ in range(len(prop_names))]
    
    for key, val in features[ps_short_name][model][frame_to_plot].items():
      filename, repeat = key
      length, manual_frs_search, true_ratings, false_ratings,\
        would_be_correct = val
      grouping, evaluation = evals[ps_short_name][model][frame_to_plot][key]
      
      denominators[0] += 1
      denominators[1] += 1
      if false_ratings is not None and len(false_ratings) > 0:
        numerators[1] += 1
        prefs = problem_prefs[ps_short_name][filename]
        for e1, e2, v in false_ratings:
          denominators[4] += 1
          if prefs[e2, e1] == v:
            numerators[4] += 1
          '''
          elif v < 1 or v > 5:
            print(model, ps_short_name, filename,
                  '%s%s%d' % (index_chars[e1], index_chars[e2], v),
                  '%d' % (prefs[e1, e2 - 1] if e2 - 1 >= 0 else -1),
                  '%d' % (prefs[e1, e2]),
                  '%d' % (prefs[e1, e2 + 1] if e2 + 1 < prefs.shape[1] else -1))
          '''
      if evaluation == 'incorrect':
        numerators[0] += 1
        denominators[2] += 1
        if false_ratings is not None and len(false_ratings) > 0:
          numerators[2] += 1
          denominators[3] += 1
          if would_be_correct == 'True':
            numerators[3] += 1
    
    xticks.append(x + (len(prop_names) - 1) / 2)
    xticklabels.append(model_short_names[model])
    
    for i in range(len(prop_names)):
      if denominators[i] > 0:
        prop, lower, upper = get_stats(numerators[i], denominators[i])
        bar_x[i].append(x)
        bar_h[i].append(prop)
        err_x.append(x)
        err_y.append(prop)
        err_lower.append(lower)
        err_upper.append(upper)
      
      x += 1
    
    x += 1
  
  for i in range(len(prop_names)):
    ax.bar(bar_x[i], bar_h[i], width=0.8, align='center',
      color=prop_colors[i], edgecolor='k', linewidth=1, label=prop_names[i])
  
  ax.errorbar(err_x, err_y, [err_lower, err_upper],
              capsize=3, color='k', linestyle='none')
  
  ax.set_xlim(-1.5, x - 0.5)
  ax.set_ylim(0, 1)
  ax.set_ylabel('Fraction of relevant events')
  ax.set_title(ps_short_name)
  ax.set_xticks(ticks=xticks, labels=xticklabels)
  
  handles, labels = ax.get_legend_handles_labels()
  for i in range(len(labels)):
    if labels[i] not in legend:
      legend[labels[i]] = handles[i]
  
  row += 1

fig.legend(handles=legend.values(), labels=legend.keys(),
           loc='upper center', ncols=5)
fig.tight_layout()
fig.subplots_adjust(top=0.89)
save_path = path.join(summary_dir, 'proportions')
plt.savefig(save_path + '.png')
plt.savefig(save_path + '.pdf')
plt.close(fig)

# Generate text file with statistics useful for tables

stats_file_path = path.join(summary_dir, 'stats.txt')
stats_file = open(stats_file_path, 'w')

for model in models_to_load:
  print('---', file=stats_file)
  print('Model: %s' % model_proper_names[model], file=stats_file)
  print('---', file=stats_file)
  print('', file=stats_file)
  
  for ps_short_name in evals:
    if model not in evals[ps_short_name]:
      continue
    
    problem_set = ps_by_short_name[ps_short_name]
    n = problem_set['n']
    k = problem_set['k']
    mean_index = (n * k - 1) / 2
    
    if 'search_for_frs' in problem_set:
      search_for_frs = problem_set['search_for_frs']
    else:
      search_for_frs = 'all'
    
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
      eval_fs = 0
      eval_wg = 0
      eval_fu = 0
      eval_incorrect_hal = 0
      eval_incorrect_hal_wouldbe = 0
      false_ratings_total = 0
      false_rating_indices = []
      false_rating_freqs = {}
      
      for key, val in evals[ps_short_name][model][frame].items():
        filename, repeat = key
        coloring, evaluation = val
        
        eval_filenames.add(filename)
        if problem_stable_counts[ps_short_name][filename] > 0:
          eval_possible_filenames.add(filename)
        
        eval_total += 1
        if evaluation == 'incorrect':
          eval_incorrect += 1
          if coloring == 'impossible':
            eval_fu += 1 # false unsolvable
          elif problem_stable_counts[ps_short_name][filename] > 0:
            eval_wg += 1 # wrong grouping
          else:
            eval_fs += 1 # false solvable
        
        if mpsf_features is not None and key in mpsf_features:
          length, manual_frs_search, true_ratings, false_ratings,\
            would_be_correct = mpsf_features[key]
          if false_ratings is not None:
            es_total += 1
            if len(false_ratings) > 0:
              es_hallucinated += 1
              prefs = problem_prefs[ps_short_name][filename]
              for rating in false_ratings:
                rating = rating[:2]
                false_ratings_total += 1
                false_rating_indices.extend(rating)
                if rating in false_rating_freqs:
                  false_rating_freqs[rating] += 1
                else:
                  false_rating_freqs[rating] = 1
              
              if evaluation == 'incorrect':
                eval_incorrect_hal += 1
                if would_be_correct == 'True':
                  eval_incorrect_hal_wouldbe += 1
      
      if search_for_frs != 'all':
        es_hallucinated = 'Unknown'
      
      if search_for_frs == 'none':
        eval_incorrect_hal = 'Unknown'
        eval_incorrect_hal_wouldbe = 'Unknown'
      
      print('Problems: %d (%d solvable)' % (len(eval_filenames),
        len(eval_possible_filenames)), file=stats_file)
      print('Trials: %s' % eval_total, file=stats_file)
      print('With false ratings: %s' % es_hallucinated, file=stats_file)
      print('Incorrect: %s' % eval_incorrect, file=stats_file)
      print('+ including false ratings: %s' % eval_incorrect_hal,
            file=stats_file)
      print('+ correct under false ratings: %s' % eval_incorrect_hal_wouldbe,
            file=stats_file)
      print('False solvable: %s' % eval_fs, file=stats_file)
      print('Wrong grouping: %s' % eval_wg, file=stats_file)
      print('False unsolvable: %s' % eval_fu, file=stats_file)
      
      if false_ratings_total > 0:
        false_rating_indices = np.array(false_rating_indices)
        mean_fr_index = np.mean(false_rating_indices)
        test_result = ttest_1samp(false_rating_indices, mean_index)
        
        print('False rating frequencies (top 10 at most):', file=stats_file)
        descending = sorted(false_rating_freqs,
                            key=lambda rating: -false_rating_freqs[rating])
        for i in range(min(len(descending), 10)):
          rating = descending[i]
          freq = false_rating_freqs[rating]
          rating_str = ''.join(index_chars[j] for j in rating)
          print('%s: %d/%d (%.3f)' % (rating_str, freq, false_ratings_total,
            freq / false_ratings_total), file=stats_file)
        
        print('Mean false rating index: %.3f (vs. %.3f, t=%.3f, p=%.6f)'\
              % (mean_fr_index, mean_index, test_result.statistic,
                 test_result.pvalue), file=stats_file)
        print('', file=stats_file)

stats_file.close()
