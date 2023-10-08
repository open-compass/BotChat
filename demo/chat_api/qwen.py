import os.path as osp
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from typing import Dict, List, Optional, Union


class QwenWrapper:
    def __init__(self, model_path: str='Qwen/Qwen-7B-Chat-Int4',system_prompt: str = None, **model_kwargs):
        self.system_prompt = system_prompt
        self.model_path=model_path
        assert osp.exists(model_path) or len(model_path.split('/')) == 2 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,device_map="auto")
        try:
            model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True,device_map="auto")
        except:
            pass
        
        model = model.eval()
        self.model = model
        self.context_length=model.config.seq_length
        self.answer_buffer = 192
        for k, v in model_kwargs.items():
            print(f'Following args are passed but not used to initialize the model, {k}: {v}. ')
        
    def length_ok(self, inputs):
        tot = 0
        for s in inputs:
            tot += len(self.tokenizer.encode(s))
        return tot + self.answer_buffer < self.context_length

    def chat(self,full_inputs: Union[str, List[str]],offset=0) -> str:
        inputs = full_inputs[offset:]
        if not self.length_ok(inputs):
            return self.chat(full_inputs, offset + 1)

        history = []
        if len(inputs) % 2 == 1:
            for i in range(len(inputs)//2):
                history.append((inputs[2*i],inputs[2*i+1]))
            input_msgs=inputs[-1]
        else:
            history.append(('',inputs[0]))
            for i in range(len(inputs)//2-1):
                history.append((inputs[2*i+1],inputs[2*i+2]))
            input_msgs=inputs[-1]

            
        response,_ = self.model.chat(self.tokenizer, input_msgs,history=history,system=self.system_prompt)
        return response,offset