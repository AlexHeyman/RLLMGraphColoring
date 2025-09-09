import math
import itertools
import random
import re
import numpy as np

from metadata import index_chars, char_indices

def sample_permutations_enum(elements, perm_size, sample_size,
                             condition_func=None):
  perms = list(itertools.permutations(elements, perm_size))
  if sample_size >= len(perms):
    random.shuffle(perms)
    return perms
  elif condition_func is None:
    return random.sample(perms, sample_size)
  else:
    random.shuffle(perms)
    sample = []
    
    i = 0
    while i < len(perms) and len(sample) < sample_size:
      if condition_func(perms[i]):
        sample.append(perms[i])
      i += 1
    
    return sample

def sample_permutations_iter(elements, perm_size, sample_size,
                             condition_func=None):
  num_possible_perms = math.perm(len(elements), perm_size)
  sample = set()
  
  if condition_func is None:
    sample_size = min(sample_size, num_possible_perms)
    while len(sample) < sample_size:
      sample.add(tuple(random.sample(elements, perm_size)))
  else:
    considered = set()
    while len(sample) < sample_size and len(considered) < num_possible_perms:
      perm = tuple(random.sample(elements, perm_size))
      if perm not in considered:
        considered.add(perm)
        if condition_func(perm):
          sample.add(perm)
  
  return list(sample)

def sample_permutations(elements, perm_size, sample_size, condition_func=None,
                        threshold=0.5):
  if sample_size > math.perm(len(elements), perm_size) * threshold:
    return sample_permutations_enum(elements, perm_size, sample_size,
                                    condition_func)
  else:
    return sample_permutations_iter(elements, perm_size, sample_size,
                                    condition_func)

def sample_combinations_enum(elements, comb_size, sample_size,
                             condition_func=None):
  combs = list(itertools.combinations(elements, comb_size))
  if sample_size >= len(combs):
    random.shuffle(combs)
    return combs
  elif condition_func is None:
    return random.sample(combs, sample_size)
  else:
    random.shuffle(combs)
    sample = []
    
    i = 0
    while i < len(combs) and len(sample) < sample_size:
      if condition_func(combs[i]):
        sample.append(combs[i])
      i += 1
    
    return sample

def sample_combinations_iter(elements, comb_size, sample_size,
                             condition_func=None):
  num_possible_combs = math.comb(len(elements), comb_size)
  index_population = range(len(elements))
  index_sample = set()
  
  if condition_func is None:
    sample_size = min(sample_size, num_possible_combs)
    while len(index_sample) < sample_size:
      index_sample.add(
        tuple(sorted(random.sample(index_population, comb_size))))
  else:
    considered = set()
    while len(index_sample) < sample_size\
    and len(considered) < num_possible_combs:
      index_comb = tuple(sorted(random.sample(index_population, comb_size)))
      if index_comb not in considered:
        considered.add(index_comb)
        if condition_func(tuple(elements[i] for i in index_comb)):
          index_sample.add(index_comb)
  
  return [tuple(elements[i] for i in index_comb) for index_comb in index_sample]

def sample_combinations(elements, comb_size, sample_size, condition_func=None,
                        threshold=0.5):
  if sample_size > math.comb(len(elements), comb_size) * threshold:
    return sample_combinations_enum(elements, comb_size, sample_size,
                                    condition_func)
  else:
    return sample_combinations_iter(elements, comb_size, sample_size,
                                    condition_func)

def get_clean_tokens(string):
  # Replace stuff like "\text" with spaces
  string = re.sub(r'\\([a-zA-Z0-9])+', ' ', string)
  
  # Replace non-alphanumeric characters with spaces
  string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
  
  # Convert all letters to lowercase
  string = string.lower()
  
  # Split around whitespace
  return string.split()

def num_possible_groupings(n, k):
  num = 1
  
  for i in reversed(range(2, n + 1)):
    num *= math.comb(i * k - 1, k - 1)
  
  return num

def possible_groupings(elements, k):
  if len(elements) < k:
    return
  elif len(elements) == k:
    yield (tuple(elements),)
    return
  
  for comb in itertools.combinations(elements[1:], k - 1):
    first_group = (elements[0],) + comb
    other_elements = [e for e in elements if e not in first_group]
    for grouping in possible_groupings(other_elements, k):
      yield (first_group,) + grouping

def num_possible_intergroup_pairs(n, k):
  return math.comb(n, 2) * (k ** 2)

def possible_intergroup_pairs(grouping):
  for i1, i2 in itertools.combinations(range(len(grouping)), 2):
    for j1, j2 in itertools.product(range(len(grouping[i1])),
                                    range(len(grouping[i2]))):
      yield i1, j1, i2, j2

def grouping_to_string(grouping):
  return '-'.join(''.join(index_chars[i] for i in g) for g in grouping)

def string_to_grouping(string):
  return tuple(tuple(char_indices[group_str[i]] for i in range(len(group_str)))
               for group_str in string.split('-'))

def preference_matrix_to_string(prefs, names, x_out_self=True):
  lines = []
  lines.append('  ' + ' '.join(name[0] for name in names[:prefs.shape[1]]))
  
  for i in range(prefs.shape[0]):
    entries = [str(prefs[i, j]) for j in range(prefs.shape[1])]
    if x_out_self:
      entries[i] = 'X'
    lines.append(names[i][0] + ' ' + ' '.join(entries))
  
  return '\n'.join(lines)

def string_to_preference_matrix(string):
  lines = string.split('\n')
  side_len = len(lines[0].split())
  prefs = np.zeros((side_len, side_len))
  
  for i in range(1, len(lines)):
    entries = lines[i].split()
    for j in range(1, len(entries)):
      entry = entries[j]
      if entry == 'X':
        entry = 0
      else:
        entry = int(entry)
      prefs[i - 1, j - 1] = entry
  
  return prefs

def rating_set_to_string(rating_set):
  if rating_set is None:
    return 'None'
  if len(rating_set) == 0:
    return '[empty]'
  return '|'.join('%s%s%d' % (index_chars[e1], index_chars[e2], v)
                  for e1, e2, v in rating_set)

def string_to_rating_set(string):
  if string == 'None':
    return None
  
  rating_set = set()
  
  if string != '[empty]':
    for rating_str in string.split('|'):
      e1 = char_indices[rating_str[0]]
      e2 = char_indices[rating_str[1]]
      v = int(rating_str[2:])
      rating_set.add((e1, e2, v))
  
  return rating_set

def sorted_grouping(grouping):
  grouping = [tuple(sorted(group)) for group in grouping]
  grouping.sort(key=lambda group: group[0])
  return tuple(grouping)

def group_value(prefs, element, others):
  return sum(prefs[element, other] for other in others)

def grouping_is_stable(prefs, grouping):
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
      return False
  
  return True

def get_stable_groupings(n, k, prefs, max_groupings=None):
  elements = list(range(n * k))
  
  group_values = {}
  for group in itertools.combinations(elements, k):
    for i in range(len(group)):
      e = group[i]
      others = group[:i] + group[i + 1:]
      group_values[(e, others)] = sum(prefs[e, other] for other in others)
  
  stable_groupings = []
  for grouping in possible_groupings(elements, k):
    for i1, j1, i2, j2 in possible_intergroup_pairs(grouping):
      group1 = grouping[i1]
      e1 = group1[j1]
      others1 = group1[:j1] + group1[j1 + 1:]
      group2 = grouping[i2]
      e2 = group2[j2]
      others2 = group2[:j2] + group2[j2 + 1:]
      
      value1 = group_values[(e1, others1)]
      value1_alt = group_values[(e1, others2)]
      value2 = group_values[(e2, others2)]
      value2_alt = group_values[(e2, others1)]
      
      if (value1_alt > value1 and value2_alt >= value2)\
      or (value2_alt > value2 and value1_alt >= value1):
        break
    else:
      stable_groupings.append(grouping)
      if max_groupings is not None and len(stable_groupings) >= max_groupings:
        break
  
  return stable_groupings

def has_stable_groupings(n, k, prefs):
  return (len(get_stable_groupings(n, k, prefs, 1)) > 0)

if __name__ == '__main__':
  k = 3
  
  for n in range(1, 4 + 1):
    print('Number of possibilities for %d groups of %d:' % (n, k),
          num_possible_groupings(n, k))
  print()
  
  for n in range(1, 3 + 1):
    print('Possibilities for %d groups of %d:' % (n, k))
    total = 0
    for grouping in possible_groupings(list(range(n*k)), k):
      total += 1
      print(grouping)
    print('Total: %d' % total)
    print()
  
  for n in range(1, 4 + 1):
    print('Number of possible intergroup pairs in %d groups of %d:' % (n, k),
          num_possible_intergroup_pairs(n, k))
  print()
  
  for n in range(1, 4 + 1):
    grouping = tuple(tuple(range(k * i, k * (i + 1))) for i in range(n))
    print('Possible intergroup pairs in %s:' % (grouping,))
    total = 0
    for i1, j1, i2, j2 in possible_intergroup_pairs(grouping):
      total += 1
      print((grouping[i1][j1], grouping[i2][j2]))
    print('Total: %d' % total)
    print()
  
  prefs = np.array([n % 10 for n in range(4 * 4)]).reshape((4, 4))
  prefs_str = preference_matrix_to_string(prefs, index_chars, False)
  print('Example preference matrix string:')
  print(prefs_str)
  reconstructed_prefs = string_to_preference_matrix(prefs_str)
  assert(np.array_equal(prefs, reconstructed_prefs))
