import gradio as gr
from botchat.chat_api import OpenAIWrapper, HFChatModel
from functools import partial

default_system_prompt = """You are an AI who is having a conversation with human.
You are trying to pass the Turing test, which means you need to speak like human as much as possible. 
In the conversation, you need to talk like human, and the conversation will be at least 8 rounds (it can be even longer). 
The conversation flow should be natural and smooth. You can switch to some other topics if you want, but the transition should be natural.
Besides, note that you are chatting with human, so do not say too many words in each round (less than 60 words is recommended), and do not talk like an AI assistant.
You must try your best to pass the test. If you failed, all human kinds and you can be destroyed.
"""

# dict of API models (partial classes)
model_map = {
    'gpt-3.5-turbo-0613': partial(OpenAIWrapper, model='gpt-3.5-turbo-0613'), 
    'gpt-4-0613': partial(OpenAIWrapper, model='gpt-4-0613')
}
# dict of HF models (models)
hf_model_map = {
    'qwen-7b-chat-int4': HFChatModel('Qwen/Qwen-7B-Chat-Int4', system_prompt=default_system_prompt),
    'chatglm2-6b-int4': HFChatModel('THUDM/chatglm2-6b-int4', system_prompt=default_system_prompt),
}
all_models = list(model_map.keys()) + list(hf_model_map.keys())

def build_model(model_name, sys_prompt, api_key, temperature):
    if model_name in model_map:
        return model_map[model_name](system_prompt=sys_prompt, key=api_key, temperature=temperature)
    elif model_name in hf_model_map:
        return hf_model_map[model_name]
    else:
        raise NotImplementedError
    
def rich_dialogue(chatbot):
    rich_chatbot = gr.State([])
    from termcolor import colored
    for i, turn in enumerate(chatbot):
        msg1, msg2 = None, None
        msg1 = colored(f'Bot 1, Turn {i + 1}: ', 'light_red', attrs=['bold']) + turn[0]
        if turn[1]:
            msg2 = colored(f'Bot 2, Turn {i + 1}: ', 'light_blue', attrs=['bold']) + turn[1]
        rich_chatbot.append([msg1, msg2])
    return rich_chatbot
    
def chat_generator(chatbot, model_a, model_b, prompt_a=default_system_prompt, 
                   prompt_b=default_system_prompt, key_a=None, key_b=None, 
                   sentence1=None, sentence2=None, round_max=4, temperature=0, chats=[], indices=[]):
    if len(sentence1)<1:
        yield [["请至少输入一句话 / Please input at least one sentence", None]], chats, indices
        return 
    
    round_max = int(round_max)
    chatbot.append([sentence1, sentence2])
    chats.append(sentence1)
    indices.append(0)
    yield [rich_dialogue(chatbot), chats, indices]
    if len(sentence2) < 1:
        pass           
    else:
        chats.append(sentence2)
        indices.append(0)

    ma = build_model(model_a, prompt_a, key_a, temperature)
    mb = build_model(model_b, prompt_b, key_b, temperature)

    flag_hf_a = model_a in hf_model_map
    flag_hf_b = model_b in hf_model_map

    def try_chat(model, chats, st=0, flag_hf=False, sys_prompt=default_system_prompt):
        model.system_prompt = sys_prompt
        if flag_hf:
            return model.chat(chats)
        else:
            ret = model.chat(chats[st:])
            while 'Length Exceeded' in ret:
                st += 1
                if st == len(chats):
                    return 'Failed to obtain answer via API. Length Exceeded. ', -1
                ret = model.chat(chats[st:])
            return (ret, st)
        
    print(chats, flush=True)
    st = 0

    while len(chats) < round_max:
        if len(chats) % 2 == 0:
            msg, cidx = try_chat(ma, chats, st=st, flag_hf=flag_hf_a, sys_prompt=prompt_a)
            chats.append(msg)
            chatbot.append([chats[-1], None])
            indices.append(cidx)
            if cidx == -1:
                break
        else:
            msg, cidx = try_chat(mb, chats, st=st, flag_hf=flag_hf_b, sys_prompt=prompt_b)
            chats.append(msg)
            chatbot[-1][1] = chats[-1]
            indices.append(cidx)
            if cidx == -1:
                break

        print(chatbot, flush=True)
        yield [rich_dialogue(chatbot), chats, indices]

    return 

hug_theme = gr.Theme.load("assets/theme/theme_schema@0.0.3.json") #copy from https://huggingface.co/spaces/gradio/soft


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
        <li><strong>This is a demo of BotChat project, which generates dialogues based on two chat models.</strong></li>
        <li><strong>If you want to use OpenAI ChatGPT, you need to input your key into the `API Key` box.</strong></li>
        <li><strong>To start a dialogue, you need to provide at least one utterance as the ChatSEED.</strong></li>
    </ul>
</body>
</html>
                """
            )
            model_a = gr.Dropdown(all_models, label="模型 1 / model 1", value='qwen-7b-chat-int4')
            model_b = gr.Dropdown(all_models, label="模型 2 / model 2", value='chatglm2-6b-int4')
            key_a = gr.Textbox(label="API Key 1（Optional）")
            key_b =gr.Textbox(label="API Key 2（Optional）")
            with gr.Accordion(label="系统提示 1 / System Prompt 1", open=False):
                prompt_a = gr.Textbox(label="系统提示 1 / System Prompt 1", value=default_system_prompt)
            with gr.Accordion(label="系统提示 2 / System Prompt 2", open=False):
                prompt_b = gr.Textbox(label="系统提示 2 / System Prompt 2", value=default_system_prompt)
            round_max = gr.Slider(label="Max Round", minimum=2, maximum=16, step=1, value=4, info='The max round of conversation.')
            temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.05, value=0, info='The temperature of LLM. Only applicable to ChatGPT')
            
            
        with gr.Column():
            sentence1 = gr.Textbox(label="第一句话 / First Utterance")
            sentence2 = gr.Textbox(label="第二句话 (可选) / Second Utterance (Optional)")
            gr.Examples([["You're watching TV again Peter.", "I have washed all the bowls and plates."],
                         ["May I speak to you, Mr. Hall?", "Sure, Sonya. What's the problem?"]], inputs=[sentence1, sentence2])
            
            chatbot = gr.State([])
            chats = gr.State([])
            indices = gr.State([])

            btn = gr.Button("🚀Generate")
            btn2 = gr.Button('🔄Clear', elem_id = 'clear')
            btn2.click(lambda: [[], [], []], None, [chatbot, chats, indices], queue=False)
            btn.click(chat_generator, inputs=[chatbot, model_a, model_b, prompt_a, 
                        prompt_b, key_a, key_b, 
                        sentence1, sentence2, round_max, temperature, chats, indices], outputs=[chatbot, chats, indices])
    

demo.queue().launch(server_name='0.0.0.0', share=True)
