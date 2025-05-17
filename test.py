'''
When run, this file takes in one or more prompts from text files (or the
console), sends them to a language model, and outputs the model's responses to
another set of text files (or the console).

Usage: python test.py <args>

Arguments:
-m, --model: The language model to use; see the function get_model_from_arg()
  for a full list of possible values.
-t, --temperature: The temperature to run the model at; 0 by default. If this
  is negative and -m uses the OpenAI API, the temperature parameter will not be
  passed to the API, to account for models like o3-mini that do not support it.
-p, --prompt_path: The path to the text file to use as a prompt, or a folder
  containing multiple prompt files (directly and/or in subfolders; the program
  will recurse). If left unspecified, the user will be asked to enter the text
  for the prompt in the console.
-r, --response_path: The path where the model's response(s) will be written in
  text file(s). Must be a directory if and only if -p is a directory. If -p and
  -r are directories, the program will create a folder structure and file names
  inside -r matching the ones in -p to put the responses in. If -r is left
  unspecified (and -p is not a directory), the response will be printed to the
  console.

The remaining arguments only have effects if -p and -r are directories.

-n, --num_processes: Number of simultaneous processes to use to send prompts
  and receive responses through the selected model's API; 1 by default.
--repeat: Number of times to send each prompt to the model, recording all of
  the responses; 1 by default. If this is greater than 1, the program will
  create an additional layer of subfolders in -r (deeper than the existing
  folders) to organize the responses.
--replace_existing: 'true' or 'false'; 'false' by default. If this is false,
  any time the program would put a response in a file that already exists, it
  will not send the prompt in the first place. Applies separately for each
  repeat of each prompt if --repeat is greater than 1.
--max_retries: If sending a prompt results in an error instead of a successful
  reception of a response, this is the number of times to retry (including the
  initial try) before the program gives up on the prompt and the applicable
  process terminates. 5 by default.
'''

import sys
from os import path, listdir, mkdir, environ
import time
import argparse
import multiprocessing as mp
import queue

import models

deepinfra_api_key = None
if 'DEEPINFRA_API_KEY' in environ:
  deepinfra_api_key = environ['DEEPINFRA_API_KEY']

xai_api_key = None
if 'XAI_API_KEY' in environ:
  xai_api_key = environ['XAI_API_KEY']

def get_model_from_arg(model_arg):
  if model_arg == 'o1-mini': # temperature must be 1
    return models.OpenAIModel('o1-mini-2024-09-12')
  elif model_arg == 'o3-mini-low': # temperature must not be specified
    return models.OpenAIModel('o3-mini-2025-01-31', reasoning_effort='low')
  elif model_arg == 'o3-mini-medium': # temperature must not be specified
    return models.OpenAIModel('o3-mini-2025-01-31', reasoning_effort='medium')
  elif model_arg == 'o3-mini-high': # temperature must not be specified
    return models.OpenAIModel('o3-mini-2025-01-31', reasoning_effort='high')
  elif model_arg == 'deepseek-r1-fireworks':
    return models.FireworksAIModel(
      'accounts/fireworks/models/deepseek-r1', max_tokens=32768)
  elif model_arg == 'deepseek-r1-deepinfra':
    return models.OpenAIModel('deepseek-ai/DeepSeek-R1',
                              base_url='https://api.deepinfra.com/v1/openai',
                              api_key=deepinfra_api_key)
  elif model_arg.startswith('claude3.7s'):
    # thinking budget must be less than max tokens
    # if thinking budget is specified, temperature must be 1
    max_tokens = 64000
    thinking_budget = -1
    
    arg_segments = model_arg.split('-')
    
    if len(arg_segments) >= 2:
      max_tokens = int(arg_segments[1])
      if len(arg_segments) >= 3:
        thinking_budget = int(arg_segments[2])
    
    return models.AnthropicModel('claude-3-7-sonnet-20250219',
      max_tokens=max_tokens, thinking_budget=thinking_budget)
  elif model_arg == 'gemini2.5pp':
    return models.GoogleModel('gemini-2.5-pro-preview-03-25')
  elif model_arg == 'grok3b':
    return models.OpenAIModel('grok-3-beta', base_url='https://api.x.ai/v1',
                              api_key=xai_api_key)
  elif model_arg == 'grok3mb-low':
    return models.OpenAIModel('grok-3-mini-beta', reasoning_effort='low',
                              base_url='https://api.x.ai/v1',
                              api_key=xai_api_key)
  elif model_arg == 'grok3mb-high':
    return models.OpenAIModel('grok-3-mini-beta', reasoning_effort='high',
                              base_url='https://api.x.ai/v1',
                              api_key=xai_api_key)
  else:
    return models.DummyModel()

def task_process(model_arg, temperature, q, pidx, max_retries):
  model = get_model_from_arg(model_arg)
  
  while not q.empty():
    try:
      task = q.get(block=True, timeout=5)
    except queue.Empty as e:
      break
    
    try:
      task_prompt_path, repeat_num, task_response_path = task
      if repeat_num < 0:
        print('Process %d: Prompting with %s' % (pidx, task_prompt_path))
      else:
        print('Process %d: Prompting with %s (repeat %d)'\
              % (pidx, task_prompt_path, repeat_num))
      
      task_prompt_file = open(task_prompt_path, 'r')
      message = task_prompt_file.read()
      task_prompt_file.close()
      
      success = False
      consecutive_tries = 1
      while not success:
        try:
          conv = model.new_conversation()
          response = conv.send_and_receive(message, temperature=temperature)
          success = True
        except Exception as e:
          print('Process %d send/receive exception: %s' % (pidx, e))
          if consecutive_tries >= max_retries:
            print('Process %d exceeded max retries; terminating' % pidx)
            return
          else:
            consecutive_tries += 1
            print('Process %d retrying (attempt %d)'\
                  % (pidx, consecutive_tries))
            time.sleep(1)
      
      task_response_file = open(task_response_path, 'w', encoding='utf-8')
      task_response_file.write(response)
      task_response_file.close()
    except Exception as e:
      print('Process %d miscellaneous exception; terminating: %s' % (pidx, e))
      return
  
  print('Process %d cannot find a new task; terminating' % pidx)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', type=str, default='dummy')
  parser.add_argument('-t', '--temperature', type=float, default=0)
  parser.add_argument('-p', '--prompt_path', type=str, default=None)
  parser.add_argument('-r', '--response_path', type=str, default=None)
  parser.add_argument('-n', '--num_processes', type=int, default=1)
  parser.add_argument('--repeat', type=int, default=1)
  parser.add_argument('--replace_existing', type=str,
                      choices=['true', 'false'], default='false')
  parser.add_argument('--max_retries', type=int, default=5)
  
  args = parser.parse_args()
  model_arg = args.model
  temperature = args.temperature
  prompt_path = args.prompt_path
  response_path = args.response_path
  num_processes = args.num_processes
  num_repeats = args.repeat
  replace_existing = (args.replace_existing == 'true')
  max_retries = args.max_retries
  
  if response_path is None:
    response_file = sys.stdout
  elif path.isdir(response_path):
    response_file = None
  else: # Response path is a file
    response_file = open(response_path, 'w', encoding='utf-8')
  
  if prompt_path is None:
    if response_file is None:
      print('Error: Response path is a directory, but prompt path is not')
      return
    
    model = get_model_from_arg(model_arg)
    conv = model.new_conversation()
    message = input('Message to send: ')
    response = conv.send_and_receive(message, temperature=temperature)
    response_file.write(response)
  elif not path.exists(prompt_path):
    print('Error: Specified prompt path does not exist')
    return
  elif path.isdir(prompt_path):
    if response_file is not None:
      print('Error: Prompt path is a directory, but response path is not')
      response_file.close()
      return
    
    if num_processes <= 0:
      print('Error: Specified number of processes is non-positive')
      return
    
    if num_repeats <= 0:
      print('Error: Specified number of repeats is non-positive')
      return
    
    tasks = []
    
    def recursively_get_tasks(relative_path):
      prompt_subdir_path = path.join(prompt_path, relative_path)
      response_subdir_path = path.join(response_path, relative_path)

      for filename in listdir(prompt_subdir_path):
        prompt_r_path = path.join(prompt_subdir_path, filename)
        response_r_path = path.join(response_subdir_path, filename)
        if path.isdir(prompt_r_path): # Prompt directory entry is a directory
          if path.exists(response_r_path):
            if not path.isdir(response_r_path):
              print('Error: %s already exists but is not a directory'\
                    % response_r_path)
              return
          else:
            mkdir(response_r_path)
          
          recursively_get_tasks(path.join(relative_path, filename))
        else: # Prompt directory entry is a file
          if num_repeats == 1:
            if replace_existing or not path.exists(response_r_path):
              tasks.append((prompt_r_path, -1, response_r_path))
          else:
            for i in range(num_repeats):
              repeat_path = path.join(response_subdir_path, 'repeat%d' % i)
              
              if path.exists(repeat_path):
                if not path.isdir(repeat_path):
                  print('Error: %s already exists but is not a directory'\
                        % repeat_path)
                  return
              else:
                mkdir(repeat_path)
              
              task_response_path = path.join(repeat_path, filename)
              if replace_existing or not path.exists(task_response_path):
                tasks.append((prompt_r_path, i, task_response_path))
    
    recursively_get_tasks('')
    
    if num_processes == 1:
      model = get_model_from_arg(model_arg)
      
      for task in tasks:
        task_prompt_path, repeat_num, task_response_path = task
        if repeat_num < 0:
          print('Prompting with %s' % task_prompt_path)
        else:
          print('Prompting with %s (repeat %d)'\
                % (task_prompt_path, repeat_num))
        
        task_prompt_file = open(task_prompt_path, 'r')
        message = task_prompt_file.read()
        task_prompt_file.close()
        
        success = False
        consecutive_tries = 1
        while not success:
          try:
            conv = model.new_conversation()
            response = conv.send_and_receive(message, temperature=temperature)
            success = True
          except Exception as e:
            print('Send/receive exception: %s' % e)
            if consecutive_tries >= max_retries:
              print('Exceeded max retries; terminating')
              return
            else:
              consecutive_tries += 1
              print('Retrying (attempt %d)' % consecutive_tries)
              time.sleep(1)
        
        task_response_file = open(task_response_path, 'w', encoding='utf-8')
        task_response_file.write(response)
        task_response_file.close()
      
      print('All tasks complete; terminating')
    else: # Spawn multiple processes to share the prompting tasks
      mp.set_start_method('spawn')

      q = mp.Queue()

      for task in tasks:
        try:
          q.put(task, block=True, timeout=5)
        except:
          print('Error: Unable to completely fill task queue')
          return

      processes = []
      
      for pidx in range(num_processes):
        process = mp.Process(target=task_process,
          args=(model_arg, temperature, q, pidx, max_retries))
        processes.append(process)
        process.start()
      
      for pidx in range(num_processes):
        processes[pidx].join()
  else: # Prompt path is a file
    if response_file is None:
      print('Error: Response path is a directory, but prompt path is not')
      return
    
    print('Prompting with %s' % prompt_path)
    
    model = get_model_from_arg(model_arg)
    conv = model.new_conversation()
    
    prompt_file = open(prompt_path, 'r')
    message = prompt_file.read()
    prompt_file.close()
    
    response = conv.send_and_receive(message, temperature=temperature)
    response_file.write(response)

  if response_file is not None:
      response_file.close()

if __name__ == '__main__':
  main()
