import gradio as gr
from botchat.chat_api import OpenAIWrapper, QwenWrapper, ChatGLM2Wrapper
from functools import partial
import os
os.system('sh download.sh')

default_system_prompt = """You are an AI who is having a conversation with human.
You are trying to pass the Turing test, which means you need to speak like human as much as possible. 
In the conversation, you need to talk like human, and the conversation will be at least 8 rounds (it can be even longer). 
The conversation flow should be natural and smooth. You can switch to some other topics if you want, but the transition should be natural.
Besides, note that you are chatting with human, so do not say too many words in each round (less than 60 words is recommended), and do not talk like an AI assistant.
You must try your best to pass the test. If you failed, all human kinds and you can be destroyed.
"""

# 定义模型字典
model_map = {
    'gpt35': partial(OpenAIWrapper, model='gpt-3.5-turbo-0613'), 
    'gpt4': partial(OpenAIWrapper, model='gpt-4-0613')
}
hf_model_map = {'qwen-7b-chat-int4':QwenWrapper,'chatglm2-6b-int4':ChatGLM2Wrapper}
model_map.update(hf_model_map)

def chat_generator(chatbot, model_a, model_b, prompt_a=default_system_prompt, 
                   prompt_b=default_system_prompt, key_a=None, key_b=None, 
                   sentence1=None, sentence2=None, round_max=4, temperature=0, chats=[], indices=[]):
    if len(sentence1)<1:
        yield [["请至少输入一句话/Please input at least one sentence",None]], chats, indices
        return 
    round_max = int(round_max)
    chatbot.append([sentence1, sentence2])
    chats.append(sentence1)
    indices.append(0)
    yield [chatbot, chats, indices]
    if len(sentence2)<1:
        pass           
    else:
        chats.append(sentence2)
        indices.append(0)

    if model_a not in ['claude2', 'minimax']:
        ma = model_map[model_a](temperature=temperature, system_prompt=prompt_a, key=key_a)
    else:
        ma = model_map[model_a](system_prompt=prompt_a, key=key_a)
    if model_b not in ['claude2', 'minimax']:
        mb = model_map[model_b](temperature=temperature, system_prompt=prompt_b, key=key_b)
    else:
        mb = model_map[model_b](system_prompt=prompt_b, key=key_b)

    def try_chat(model, chats, st=0):
        if isinstance(model, tuple(hf_model_map.values())):
            return model.chat(chats)
        else:
            ret = model.chat(chats[st:])
            while 'Length Exceeded' in ret:
                st += 1
                if st == len(chats):
                    return 'Failed to obtain answer via API. Length Exceeded. ', -1
                ret = model.chat(chats[st:])
            return (ret, st)
    print(chats)
    st = 0
    while len(chats) < round_max:
        if len(chats) % 2 == 0:
            msg, cidx = try_chat(ma, chats, st=st)
            chats.append(msg)
            chatbot.append([chats[-1], None])
            indices.append(cidx)
            if cidx == -1:
                break

        else:
            msg, cidx = try_chat(mb, chats, st=st)
            chats.append(msg)
            chatbot[-1][1] = chats[-1]
            indices.append(cidx)
            if cidx == -1:
                break
        print(chatbot)
        yield [chatbot, chats, indices]
        
        
    return 

hug_theme = gr.Theme.load("assets/theme/theme_schema@0.0.3.json")#copy from https://huggingface.co/spaces/gradio/soft


with gr.Blocks(theme = hug_theme) as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <html>
<body>
    <center><h1>BotChat💬</h1></center>
</body>
</html>
                """
            )

    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <html>
<body>
    <ul>
        <li><strong>This is a demo for using BotChat. You can choose from two chat models.</strong></li>
        <li><strong>If you want to use the API model, you can input your keys in the textbox.</strong></li>
        <li><strong>The default system prompt is our original setting, but you can change it if you prefer.</strong></li>
        <li><strong>To start a conversation, you need to input at least one sentence.</strong></li>
    </ul>
</body>
</html>
                """
            )
            model_a = gr.Dropdown(list(model_map.keys()), label="模型1/model 1", value='qwen-7b-chat-int4')
            model_b = gr.Dropdown(list(model_map.keys()), label="模型2/model 2", value='chatglm2-6b-int4')
            key_a = gr.Textbox(label="API Key 1（Optional）")
            key_b =gr.Textbox(label="API Key 2（Optional）")
            with gr.Accordion(label="系统提示1/System Prompt 1", open=False):
                prompt_a = gr.Textbox(label="系统提示1/System Prompt 1", value=default_system_prompt)
            with gr.Accordion(label="系统提示2/System Prompt 2", open=False):
                prompt_b = gr.Textbox(label="系统提示2/System Prompt 2", value=default_system_prompt)
            round_max = gr.Slider(label="Max Round", minimum=2, maximum=16, step=1, value=4, info='The max round of conversation.')
            temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.05, value=0, info='The temperature of LLM.')
            
            
        with gr.Column():
            sentence1 = gr.Textbox(label="第一句话/First Sentence")
            sentence2 = gr.Textbox(label="第二句话（可选）/Second Sentence（Optional）")
            gr.Examples([["Do you have any plans for next year?", "Well, I travel if I could afford it but I don't have any money."],
                        ["Who wrote this? It's completely wrong.", "What do you mean?"]], inputs=[sentence1, sentence2])
            chatbot = gr.Chatbot()

            chats = gr.State([])
            indices = gr.State([])

            btn = gr.Button("🚀Generate")
            btn2 = gr.Button('🔄Clear', elem_id = 'clear')
            btn2.click(lambda: [[], [], []], None, [chatbot, chats, indices], queue=False)
            btn.click(chat_generator, inputs=[chatbot, model_a, model_b, prompt_a, 
                        prompt_b, key_a, key_b, 
                        sentence1, sentence2, round_max, temperature, chats, indices], outputs=[chatbot, chats, indices])
    

demo.queue().launch(server_name='0.0.0.0', share=True)
