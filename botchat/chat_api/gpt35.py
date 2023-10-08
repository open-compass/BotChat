import openai
import time
from typing import Dict, List, Optional, Union
from collections import defaultdict



class OpenAIWrapper:

    is_api: bool = True

    def __init__(self, 
                 model: str = 'gpt-3.5-turbo-0613', 
                 retry: int = 8,
                 wait: int=5, 
                 verbose: bool = False, 
                 system_prompt: str = None,
                 temperature: float = 0,
                 key: str = None,
                ):
        
        import tiktoken
        self.tiktoken = tiktoken

        self.model = model
        self.system_prompt = system_prompt
        self.retry = retry
        self.wait = wait
        self.cur_idx = 0
        self.fail_cnt = defaultdict(lambda: 0)
        self.fail_msg = 'Failed to obtain answer via API. '
        self.temperature = temperature
        self.keys = [key]
        self.num_keys = 1
        self.verbose = verbose
    
    
    def generate_inner(self,
                       inputs: Union[str, List[str]],
                       max_out_len: int = 1024,
                       chat_mode=False, 
                       temperature: float = 0) -> str:
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        if isinstance(inputs, str):
            input_msgs.append(dict(role='user', content=inputs))
        elif self.system_prompt is not None and isinstance(inputs, list) and len(inputs) == 0:
            pass   
        else:
            assert isinstance(inputs, list) and isinstance(inputs[0], str)
            if chat_mode:
                roles = ['user', 'assistant'] if len(inputs) % 2 == 1 else ['assistant', 'user']
                roles = roles * len(inputs)
                for role, msg in zip(roles, inputs):
                    input_msgs.append(dict(role=role, content=msg))
            else:
                for s in inputs:
                    input_msgs.append(dict(role='user', content=s))

        for i in range(self.num_keys):
            idx = (self.cur_idx + i) % self.num_keys
            if self.fail_cnt[idx] >= min(self.fail_cnt.values()) + 20:
                continue
            try:
                openai.api_key = self.keys[idx]
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=input_msgs,
                    max_tokens=max_out_len,
                    n=1,
                    stop=None,
                    temperature=temperature,)
                
                result = response.choices[0].message.content.strip()
                self.cur_idx = idx
                return result
            except:
                print(f'OPENAI KEY {self.keys[idx]} FAILED !!!')
                self.fail_cnt[idx] += 1
                if self.verbose:
                    try:
                        print(response)
                    except:
                        pass

                pass
            
        x = 1 / 0

    def chat(self, inputs, max_out_len=1024, temperature=0):

        if isinstance(inputs, str):
            context_window = 4096
            if '32k' in self.model:
                context_window = 32768
            elif '16k' in self.model:
                context_window = 16384
            elif 'gpt-4' in self.model:
                context_window = 8192
            # Will hold out 200 tokens as buffer
            max_out_len = min(max_out_len, context_window - self.get_token_len(inputs) - 200)
            if max_out_len < 0:
                return self.fail_msg + 'Input string longer than context window. Length Exceeded. '
            
        assert isinstance(inputs, list)
        for i in range(self.retry):
            try:
                return self.generate_inner(inputs, max_out_len, chat_mode=True, temperature=temperature)
            except:
                if i != self.retry - 1:
                    if self.verbose:
                        print(f'Try #{i} failed, retrying...')
                    time.sleep(self.wait)
                pass
        return self.fail_msg
        
   
