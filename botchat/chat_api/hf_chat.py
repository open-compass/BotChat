import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
import copy as cp
import os.path as osp
import torch.nn as nn
from importlib import reload
import torch

def get_gpu_num(model_name):
    model_name = model_name.lower()
    kws = {
        8: ['65b', '70b'],
        4: ['30b', '33b', '35b', '40b'],
        2: ['13b', '14b', '20b'],
        1: ['6b', '7b'],
    }
    for k in [8, 4, 2, 1]:
        for keyword in kws[k]:
            if keyword in model_name:
                return k
    return 8

model_map = {
    'internlm-chat-7b': 'internlm/internlm-chat-7b',
    'internlm-chat-20b': 'internlm/internlm-chat-20b',
    'qwen-7b-chat': 'Qwen/Qwen-7B-Chat',
    'chatglm2-6b': 'THUDM/chatglm2-6b',
    'baichuan2-13b-chat': 'baichuan-inc/Baichuan2-13B-Chat', 
    'qwen-14b-chat': 'Qwen/Qwen-14B-Chat', 
    'vicuna-13b-v1.5':'lmsys/vicuna-13b-v1.5',
    'vicuna-7b-v1.5':'lmsys/vicuna-7b-v1.5'
}
Auto_model = [model_map['chatglm2-6b']]

class HFChatModel:

    def _get_context_length(self, model, model_path):
        # By default, we use model.config.seq_length
        model_path = model_path.lower()
        if 'baichuan' in model_path:
            context_window = model.config.model_max_length
        elif 'internlm' in model_path:
            context_window = model.config.max_position_embeddings
        elif 'vicuna' in model_path:
            context_window = model.generation_config.max_length
        else:
            # chatglm & qwen
            context_window = model.config.seq_length
        return context_window 
    
    def __init__(self, 
                 model_path, 
                 system_prompt: str=None,
                 temperature: float=0, 
                 **model_kwargs):
        
        if 'vicuna' in model_path.lower():
            try:
                from fastchat.model import get_conversation_template
            except:
                warnings.warn("Please install fastchat first to use vicuna. ")

        self.explicit_device = model_kwargs.pop('device', None)

        if self.explicit_device is None:
            # If CUDA_VISIBLE_DEVICES is not properly set
            if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ['CUDA_VISIBLE_DEVICES'] in ['', '0,1,2,3,4,5,6,7']:
                num_gpu = get_gpu_num(model_path)
                gpu_offset = model_kwargs.pop('gpu_offset', 0)
                cuda_visible_devices = ','.join([str(i) for i in range(gpu_offset, gpu_offset+num_gpu)])
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
        from transformers.generation import GenerationConfig
        
        if model_path in model_map:
            model_path = model_map[model_path]
        self.model_path=model_path
        if model_path in Auto_model:
            LoadModel=AutoModel
        else:
            LoadModel=AutoModelForCausalLM
        assert osp.exists(model_path) or len(model_path.split('/')) == 2

        device = self.explicit_device if self.explicit_device else "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = LoadModel.from_pretrained(model_path, trust_remote_code=True, device_map='cpu')
        if device != 'cpu':
            model = model.to(f'cuda:{device}' if isinstance(device, int) else 'cuda')
        try:
            model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True, device_map=device)
        except:
            pass

        torch.cuda.empty_cache()
        self.model = model.eval()
        self.context_length = self._get_context_length(model=model, model_path=model_path)
        self.answer_buffer = 192
        self.system_prompt = system_prompt
        self.temperature = temperature
        if temperature > 0:
            warnings.warn('Temperature is not supported for HFChatModel.')
        for k, v in model_kwargs.items():
            warnings.warn(f'Following args are passed but not used to initialize the model, {k}: {v}. ')
        
    def generate(self, input):
        if 'baichuan' in self.model_path.lower():
            messages=[]
            messages.append({"role": "user", "content": input})
            resp= self.model.chat(self.tokenizer, messages)
        elif 'vicuna' in self.model_path.lower():
            from fastchat.model import get_conversation_template
            conv = get_conversation_template('vicuna')
            conv.append_message(conv.roles[0], input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = self.tokenizer([prompt], return_tensors="pt")
            if torch.cuda.is_available():
                for k in inputs:
                    inputs[k] = inputs[k].cuda()
            outputs = self.model.generate(**inputs, do_sample=True, temperature=0.7, repetition_penalty=1.0, max_new_tokens=512)
            resp = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True, spaces_between_special_tokens=False)

        else:
            resp, _ = self.model.chat(self.tokenizer, input, history=[])

        return resp

    def length_ok(self, inputs):
        tot = len(self.tokenizer.encode(self.system_prompt)) if self.system_prompt is not None else 0
        for s in inputs:
            tot += len(self.tokenizer.encode(s))
        return tot + self.answer_buffer < self.context_length
    
    def chat(self, full_inputs, offset=0):
        assert isinstance(full_inputs, list)

        inputs = full_inputs[offset:]
        if not self.length_ok(inputs):
            return self.chat(full_inputs, offset + 1)
        
        model_path = self.model_path.lower()
        
        if sum([x in model_path for x in ['baichuan']]):
            input_msgs = []
            if self.system_prompt is not None:
                input_msgs.append(dict(role='user', content=self.system_prompt))
            if len(inputs):
                assert isinstance(inputs, list) and isinstance(inputs[0], str)
                roles = ['user', 'assistant'] if len(inputs) % 2 == 1 else ['assistant', 'user']
                roles = roles * len(inputs)
                for role, msg in zip(roles, inputs):
                    input_msgs.append(dict(role=role, content=msg))
            response = self.model.chat(self.tokenizer, input_msgs)
        elif sum([x in model_path for x in ['vicuna']]):
            from fastchat.model import get_conversation_template
            conv = get_conversation_template('vicuna')
            assert isinstance(inputs, list) and isinstance(inputs[0], str)
            if len(inputs) % 2 == 1:
                if self.system_prompt is not None:
                    conv.append_message(conv.roles[0], self.system_prompt)
                for i in range(len(inputs)//2):
                    conv.append_message(conv.roles[0], inputs[2 * i])
                    conv.append_message(conv.roles[1], inputs[2 * i + 1])
            else:
                assert self.system_prompt is not None
                conv.append_message(conv.roles[0], self.system_prompt)
                conv.append_message(conv.roles[1], inputs[0])
                for i in range(len(inputs) // 2 - 1):
                    conv.append_message(conv.roles[0], inputs[2 * i + 1])
                    conv.append_message(conv.roles[1], inputs[2 * i + 2])
            conv.append_message(conv.roles[0], inputs[-1])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = self.tokenizer([prompt], return_tensors="pt")
            if torch.cuda.is_available():
                for k in inputs:
                    inputs[k] = inputs[k].cuda()
            outputs = self.model.generate(**inputs, do_sample=True, temperature=0.7, repetition_penalty=1.0, max_new_tokens=512)
            response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True, spaces_between_special_tokens=False)
            response = response.lstrip('\n')
        else:
            # The default option, support internlm, chatglm, qwen
            history, msg = [], None
            if len(inputs) % 2 == 1:
                if self.system_prompt is not None:
                    history = [(self.system_prompt, '')]
                for i in range(len(inputs)//2):
                    history.append((inputs[2 * i], inputs[2 * i + 1]))    
            else:
                assert self.system_prompt is not None
                history = [(self.system_prompt, inputs[0])]
                for i in range(len(inputs) // 2 - 1):
                    history.append((inputs[2 * i + 1], inputs[2 * i + 2]))
            msg = inputs[-1]
            response, _ = self.model.chat(self.tokenizer, msg, history=history)
        
        return response, offset