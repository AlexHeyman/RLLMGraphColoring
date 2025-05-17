from os import environ
import sys
import httpx
# from meta_ai_api import MetaAI
import fireworks.client
from openai import OpenAI
import anthropic
import google.generativeai as genai


if 'FIREWORKSAI_API_KEY' in environ:
  fireworks.client.api_key = environ['FIREWORKSAI_API_KEY']

if 'GOOGLE_API_KEY' in environ:
  genai.configure(api_key=environ['GOOGLE_API_KEY'])


class LanguageModel:
  
  def new_conversation(self):
    raise NotImplementedError


class Conversation:
  
  def send_and_receive(self, message, temperature=0):
    raise NotImplementedError


class DummyModel(LanguageModel):
  
  def __init__(self):
    pass
  
  def new_conversation(self):
    return DummyModelConversation()


class DummyModelConversation(Conversation):
  
  def __init__(self):
    pass
  
  def send_and_receive(self, message, temperature=0):
    return 'Dummy model response'


class FireworksAIModel(LanguageModel):
  
  def __init__(self, model_name, max_tokens=4096):
    self.model_name = model_name
    self.max_tokens = max_tokens
  
  def new_conversation(self):
    return FireworksAIModelConversation(self)


class FireworksAIModelConversation(Conversation):
  
  def __init__(self, model):
    self.model = model
    self.has_sent = False
  
  def send_and_receive(self, message, temperature=0):
    if self.has_sent:
      raise RuntimeError(
          'FireworksAIModelConversation does not support multiple messages')
    self.has_sent = True
    
    response = fireworks.client.ChatCompletion.create(
      self.model.model_name,
      messages=[{'role': 'user', 'content': message}],
      temperature=temperature,
      n=1,
      max_tokens=self.model.max_tokens
    )
    return response.choices[0].message.content


class OpenAIModel(LanguageModel):
  
  def __init__(self, model_name, base_url=None, api_key=None, timeout=600,
               reasoning_effort=None):
    self.client = OpenAI(base_url=base_url, api_key=api_key,
                         timeout=httpx.Timeout(timeout))
    self.model_name = model_name
    self.reasoning_effort = reasoning_effort
  
  def new_conversation(self):
    return OpenAIModelConversation(self)


class OpenAIModelConversation(Conversation):
  
  def __init__(self, model):
    self.model = model
    self.has_sent = False
  
  def send_and_receive(self, message, temperature=0):
    if self.has_sent:
      raise RuntimeError(
          'OpenAIModelConversation does not support multiple messages')
    self.has_sent = True
    
    args = dict(model=self.model.model_name,
                messages=[{'role': 'user', 'content': message}]
                )
    
    if temperature >= 0:
      args['temperature'] = temperature
    
    if self.model.reasoning_effort is not None:
      args['reasoning_effort'] = self.model.reasoning_effort
    
    try:
      response = self.model.client.chat.completions.create(**args)
      message = response.choices[0].message
      output_blocks = []
      
      if hasattr(message, 'reasoning_content'):
        output_blocks.append(['<think>', message.reasoning_content, '</think>'])
      
      output_blocks.append([message.content])
      
      return '\n\n'.join('\n'.join(block) for block in output_blocks)
    except httpx.TimeoutException as e:
      raise RuntimeError('Response timeout')


class AnthropicModel(LanguageModel):
  
  def __init__(self, model_name, timeout=600, max_tokens=4096,
               thinking_budget=-1):
    self.client = anthropic.Anthropic(timeout=httpx.Timeout(timeout))
    self.model_name = model_name
    self.max_tokens = max_tokens
    self.thinking_budget = thinking_budget
  
  def new_conversation(self):
    return AnthropicModelConversation(self)


class AnthropicModelConversation(Conversation):
  
  def __init__(self, model):
    self.model = model
    self.has_sent = False
  
  def send_and_receive(self, message, temperature=0):
    if self.has_sent:
      raise RuntimeError(
          'AnthropicModelConversation does not support multiple messages')
    self.has_sent = True
    
    args = dict(model=self.model.model_name,
                max_tokens=self.model.max_tokens,
                temperature=temperature,
                messages=[{'role': 'user',
                           'content': [{'type': 'text', 'text': message}]
                           }]
                )
    
    if self.model.thinking_budget >= 0:
      args['thinking'] = {'type': 'enabled',
                          'budget_tokens': self.model.thinking_budget}
    
    try:
      response = self.model.client.messages.create(**args)
      output_blocks = []
      
      for block in response.content:
        if block.type == 'text':
          output_blocks.append([block.text])
        elif block.type == 'thinking':
          output_blocks.append(['<think>', block.thinking, '</think>'])
        elif block.type == 'redacted_thinking':
          output_blocks.append(['<redacted_think>', block.data,
                                '</redacted_think>'])
      
      return '\n\n'.join('\n'.join(block) for block in output_blocks)
    except anthropic.InternalServerError as e:
      raise RuntimeError(str(e))


class GoogleModel(LanguageModel):
  
  def __init__(self, model_name):
    self.client = genai.GenerativeModel(model_name)
    self.model_name = model_name
  
  def new_conversation(self):
    return GoogleModelConversation(self)


class GoogleModelConversation(Conversation):
  
  def __init__(self, model):
    self.model = model
    self.has_sent = False
  
  def send_and_receive(self, message, temperature=0):
    if self.has_sent:
      raise RuntimeError(
          'GoogleModelConversation does not support multiple messages')
    self.has_sent = True
    
    response = self.model.client.generate_content(message,
      generation_config=genai.GenerationConfig(temperature=temperature))
    
    return response.text
