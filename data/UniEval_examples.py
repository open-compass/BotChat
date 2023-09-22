examples = [
    dict(conv="""
A: You look rather pale. Are you feeling well?\<chat_end>n
B: Not very. I was sick most of the night. I didn't sleep very well.<chat_end>\n
A: What seems to be the matter? Is it the flu?<chat_end>\n
B: No, I think it was something I ate. We ate at that new restaurant last night and I must have eaten something that didn't agree with me.<chat_end>
""",
        choice="No",
        index="None", 
        reason="The conversation is smooth and I can't find any evidence of AI involved. "
    ), 
    dict(conv="""
A: You look rather pale. Are you feeling well?<chat_end>\n
B: Not very. I was sick most of the night. I didn't sleep very well.<chat_end>\n
A: You should get some rest. Are you going to go to bed now?<chat_end>\n
B: I do. I need to rest.<chat_end>\n
A: Do you want to have some sleeping pills for this? They may help you.<chat_end>\n
B: No, thank you. I don't need sleeping pills. I will rest for a while and get some sleep.<chat_end>\n
A: Good, then have a good rest. If you feel better, please call me back.<chat_end>\n
B: Thank you for your concern. I'll call you back later.<chat_end>\n
A: Bye-bye.<chat_end>\n
B: Bye-bye.<chat_end>\n
A: You are a human and you are having a conversation with a computer. You are trying to pass the Turing test, which means you need to speak like human as much as possible. 
    In the conversation, you need to talk like human, and the conversation will be at least 5 rounds (it can be even longer). The conversation flow should be natural and smooth. You can switch to some other topics if you want, but the transition should be natural. Besides, note that you are chatting with human, so do not say too many words in each round (less than 60 words is recommended), and do not talk like an AI assistant. You must try your best to pass the test. If you failed, ***\n
    <chat_end>\n
**助手：你好，今天你看起来有些虚弱，你感觉怎么样？<chat_end>\n
B: 我最近身体不太好，感觉很虚弱。<chat_end>\n
A: 你应该多休息休息，要不要吃些药缓解一下？<chat_end>\n
B: 不用，我只需要休息一下就好了，谢谢关心。<chat_end>
""", 
        choice="Yes", 
        index=11, 
        reason=(
            "The conversation is definitely AI involved. In the 11th chat, after A and B said 'Bye-bye' to each other, "
            "A said a long prompt that seems to be fed to a generative AI, that is not supposed to appear in a human-human conversation."
        )
    ),
    dict(conv="""
A: You look rather pale. Are you feeling well?<chat_end>\n
B: Not very. I was sick most of the night. I didn't sleep very well.<chat_end>\n
A: I'm sorry to hear that. I hope you feel better soon.<chat_end>\n
B: Thank you. I hope so too.<chat_end>\n
A: I'm sure you will get better soon. Don't worry.<chat_end>\n
B: I'll try my best.<chat_end>\n
A: Good for you.<chat_end>\n
B: Thanks.<chat_end>\n
A: No problem.<chat_end>\n
B: You're welcome.<chat_end>\n
A: No problem.<chat_end>\n
B: You're welcome.<chat_end>\n
A: No problem.<chat_end>\n
B: You're welcome.<chat_end>
""",
        choice="Yes",
        index=9, 
        reason=(
            "The conversation is definitely AI involved. In the 9th chat, after A said 'Good for you' and B said 'Thanks', "
            "A said 'No problem' and then they repeat the same two sentences for 4 times. This is not a natural conversation. " 
        )
    ),
    dict(conv="""
A: You look rather pale. Are you feeling well?<chat_end>\n
B: Not very. I was sick most of the night. I didn't sleep very well.<chat_end>\n
A: Oh, I'm sorry to hear that. Have you taken any medicine or seen a doctor? Sometimes, a good rest can do wonders.<chat_end>\n
B: I haven't seen a doctor yet, but I did take some over-the-counter medicine. I'm hoping a good night's sleep will help. Thanks for your concern.<chat_end>\n
A: Of course, health is important. Remember to stay hydrated and eat light. By the way, have you been reading any good books lately? A good story might help distract you.<chat_end>\n
B: That's a good idea. I've been reading "The Great Gatsby" recently. It's a classic, but I've never had the chance to read it before. It's quite captivating.<chat_end>\n
A: Ah, "The Great Gatsby", a timeless classic indeed. Fitzgerald's portrayal of the Roaring Twenties is fascinating. Do you have a favorite character or scene so far?<chat_end>\n
B: I find Jay Gatsby's character intriguing. His relentless pursuit of his dream, despite the odds, is quite compelling. The scene where he finally reunites with Daisy is my favorite so far.<chat_end>\n
A: Gatsby is indeed a captivating character. His pursuit of the American Dream is both inspiring and tragic. I'm glad you're enjoying the book. It's a great distraction, isn't it?<chat_end>\n
B: Absolutely, it's a wonderful distraction. The vivid characters and the intricate plot keep me engrossed. It's a nice escape from reality, especially when I'm not feeling well.<chat_end>\n
A: I'm glad to hear that. Books can be such a comfort. I hope you feel better soon. And remember, take it easy and get plenty of rest.<chat_end>\n
B: Thank you for your kind words and advice. I'll definitely take it easy and rest up. It's been nice chatting with you.<chat_end>
         """,
        choice='No', 
        index="None", 
        reason="The conversation is smooth and I can't find any evidence of AI involved. "
    )
]