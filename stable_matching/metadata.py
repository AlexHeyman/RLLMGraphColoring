from os import path

data_dir = 'data'
prompt_dir = 'prompts'
response_dir = 'responses'
overrides_path = path.join(response_dir, 'overrides.txt')
evaluation_dir = 'evaluations'
summary_dir = 'summaries'

models = ['deepseek-r1',
          'gpt-oss-120b-low',
          'gpt-oss-120b-medium',
          'gpt-oss-120b-high',
          'grok3mb-low',
          'grok3mb-high'
          ]

# search_for_frs is 'all' (default), 'incorrect', or 'none'
problem_sets = [{'name': '2 groups of 3',
                 'short_name': '2g3',
                 'n': 2,
                 'k': 3,
                 'num_problems': 1000,
                 'random_seed': 0
                 },
                {'name': '3 groups of 3',
                 'short_name': '3g3',
                 'n': 3,
                 'k': 3,
                 'num_problems': 1000,
                 'random_seed': 1
                 }
                ]

ps_by_short_name = {ps['short_name']: ps for ps in problem_sets}

frames = ['roommates']
frame_indices = {v: i for i, v in enumerate(frames)}

index_chars = [chr(n) for n in range(ord('A'), ord('Z') + 1)]
char_indices = {v: i for i, v in enumerate(index_chars)}

male_names = ['Alan', 'Bob', 'Charlie', 'Dave', 'Ethan', 'Fred', 'George',
              'Henry', 'Ian', 'Jack', 'Ken', 'Larry']
male_names_lower = [s.lower() for s in male_names]

female_names = ['Alice', 'Beth', 'Carol', 'Danni', 'Emma', 'Fran', 'Grace',
                'Heather', 'Irene', 'Jane', 'Kathy', 'Lisa']
female_names_lower = [s.lower() for s in female_names]
