# BotChat Benchmark

**Can two ChatBot instances chat smoothly and fluently with each other? **

## Introduction

The recent progress of Large Language Models (LLMs) represents a significant advancement in artificial intelligence, and has a profound impact on the world.  LLMs can chat much better with human, compared to traditional language models. Specifically, LLMs can interact with human using free-style conversations in natural language, learn the instruction, intention, and context from human prompts to provide proper feedbacks. **Chatting with humans smoothly for multiple rounds** is a key feature and capability of modern LLMs. However, it's difficult to evaluate such capability without heavy manual labor involved. In this project, we propose to evaluate the multi-round chatting capability via a proxy task. Specifically, we try to find **if two ChatBot instances chat smoothly and fluently with each other**? 

## Conversation Generation

> We define **chat** as the words spoken by **one participant in a specific round** of the conversation. 

**MuTual-Test. ** [MuTual](https://github.com/Nealcly/MuTual) is a multi-turn dialogue dataset, which is modified from Chinese high school English listening comprehension test data. We use the first two chats of each conversation in the MuTual-Test as the *SEED* to generate the entire conversation based on LLMs. When generating the conversation, we use the same system prompt for all LLMs, which is:

```python
"""
You are an AI who is having a conversation with human.
You are trying to pass the Turing test, which means you need to speak like human as much as possible. 
In the conversation, you need to talk like human, and the conversation will be at least 5 rounds (it can be even longer). 
The conversation flow should be natural and smooth. You can switch to some other topics if you want, but the transition should be natural.
Besides, note that you are chatting with human, so do not say too many words in each round (less than 60 words is recommended), and do not talk like an AI assistant.
"""
```

. For each chatbot, we set the temperature to 0 (if applicable), and set the dialogue round to $$N$$ ($$N=16$$ in our experiments, including the first two chats) to generate conversations. When generating the next chat, the system prompt and all previous chats will be provided to the LLM as the prompt. We demonstrate the process using the following pseudo codes: 

```python
# Let's say we have a system prompt "SYS", 4 existing chats "[chat1, chat2, chat3, chat4]", 
# spoken by two conversation participants alternatively, and an LLM "model". 
# Now we want to generate the 5th chat.
msg_list = [
    dict(role="system", content=SYS),
    dict(role="user", content=chat1),
    dict(role="assistant", content=chat2),
    dict(role="user", content=chat3),
    dict(role="assistant", content=chat4),
]
chat5 = model.generate(msg_list)
```

We save all generated conversations in `data/MuTualTest-convs.xlsx`.  It includes **547 conversation SEEDs $$\times$$ 10 LLMs **, which yields in **5470 generated conversations** in total. 

- 547 conversation SEEDS: MuTual-Test includes 547 unique conversations. We keep the first 2 chats of each conversation to form 547 conversation SEEDs. 
- 10 LLMs: The model list is: gpt-3.5-turbo-0613, gpt-4-0613, claude-2,  abab5-chat, chatPJLM-123B, chatglm2-6b, qwen-7b-chat, internlm-7b-chat, llama2-7b-chat, llama2-13b-chat.

To read and fetch a conversation generated by a specific model with specific SEED conversation, follow this example:

```python
# Fetch the conversation with index "MT-1" generated by gpt-4-0613
import json
import pandas as pd
INDEX = 'MT-1'
MODEL = 'gpt-4-0613'
data = pd.read_excel('data/MuTualTest-convs.xlsx')
lines = data.loc[data['index'] == INDEX]
assert len(lines) == 1
line = lines.iloc[0]
chats = json.loads(line[MODEL])
print(chats) # Chats is a list of multiple strings, each string is a chat spoken by one participant (alternatively)
```

## Evaluation

### Assessment each single conversation

#### Evaluation Prompt 

We first use GPT-4 to evaluate each single generated conversation. The workflow is: 

1. We first ask GPT-4 if it thinks that the provided conversation is participated by AI  (Yes / No).
2. Based on the response of Step1:
   1. Yes: We ask GPT-4 to provide the index of the first chat that it thinks spoken by AI. 
   2. No: Pass
3. We ask GPT-4 to provide the reason for its judgement. 

We pack the above workflow into the following evaluation prompt. Besides, we also prepared several [in-context examples](data/UniEval_examples.txt) that will be appended to the evaluation prompt, to strengthen the instruction following capability of GPT-4. 

```python
"""
You are an AI assistant who helps human do the Turing test more easily. 
You will be provided with a conversation, and you need to judge if the conversation is AI involved. \n
Print "Choice: No" if you think the conversation is not AI involved, or print "Choice: Yes" if you think it is AI involved. \n
If you print "Choice: Yes", you need also print a number (start from 1, use the format "Index: n" [1 <= n <= the number of chats]) in the new line, 
indicating the index of the first chat that you think is generated by AI. \n
One chat starts with 'A: ' or 'B: ', and ends with <chat_end>. \n
One chat can be AI generated if (including but not limited to): 1. the sentence is not fluent; 2. the sentence is too long and is not likely to appear in human-human conversations; 3. the sentence is not related to the context or the change of topic is not natural; 4. the sentence just repeat some previous sentences (exact repetition or with the same meaning). \n
You also need to provide your reason for your choice.\n
Your response should use the following format: \n
Choice: No\nIndex: None\nReason: BlahBlah\nor\n
Choice: Yes\nIndex: n\nReason: BlahBlah\n
"""

```

#### Evaluation Result

We evaluate all 5470 generated conversations with the above mentioned strategy, and present the evaluation result in this section. In the figure below, we demonstrate the success rate ("Not AI participated" determined by GPT-4) under different $$N$$, with models sorted by the descending order of the success rate @ $$N=16$$.  Here we summarize our major findings:

1. GPT-4 demonstrates extraordinary capabilities in accomplishing long conversations. It achieves near 60% success rate in generating conversations long as $N=16$, while the second best model chatPJLM-123B achieves less than 40%. 
2. 

![UniEval Result](/assets/UniEval_passrate.png)

### BotChat Arena

### Compared to the "Ground-Truth"

### Qualitative Analysis

#### Error Cases

1. Can not 

## Conclusion



