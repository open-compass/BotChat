import os.path as osp
from transformers import AutoTokenizer, AutoModel
from transformers.generation import GenerationConfig
from typing import Dict, List, Optional, Union


class ChatGLM2Wrapper:
    def __init__(self, 
                 model_path: str = 'THUDM/chatglm2-6b-int4',
                 system_prompt: str = None,
                 temperature: float = 0,
                 **model_kwargs):
        
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model_path=model_path
        assert osp.exists(model_path) or len(model_path.split('/')) == 2
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        try:
            self.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
            self.model.generation_config = self.generation_config
        except:
            pass
        
        self.model = self.model.eval()
        self.context_length = self.model.config.seq_length
        self.answer_buffer = 192
        for k, v in model_kwargs.items():
            print(f'Following args are passed but not used to initialize the model, {k}: {v}. ')

    def length_ok(self, inputs):
        tot = len(self.tokenizer.encode(self.system_prompt)) if self.system_prompt is not None else 0
        for s in inputs:
            tot += len(self.tokenizer.encode(s))
        return tot + self.answer_buffer < self.context_length

        
    def chat(self, full_inputs: Union[str, List[str]], offset=0) -> str:
        inputs = full_inputs[offset:]
        if not self.length_ok(inputs):
            return self.chat(full_inputs, offset + 1)

        history_base, history, msg = [], [], None
        if len(inputs) % 2 == 1:
            if self.system_prompt is not None:
                history_base = [(self.system_prompt, '')]
            for i in range(len(inputs)//2):
                history.append((inputs[2 * i], inputs[2 * i + 1]))
            msg = inputs[-1]
        else:
            assert self.system_prompt is not None
            history_base = [(self.system_prompt, inputs[0])]
            for i in range(len(inputs) // 2 - 1):
                history.append((inputs[2 * i + 1], inputs[2 * i + 2]))
            msg = inputs[-1]
            
        response, _ = self.model.chat(self.tokenizer, msg, history=history_base + history, do_sample=False, temperature=self.temperature)
        return response, offset
