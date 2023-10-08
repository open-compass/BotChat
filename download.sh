#!/bin/sh
if [ ! -d "llm_weights/chatglm2-6b-int4" ]
then
    echo "Downloading..."
    git lfs clone https://huggingface.co/THUDM/chatglm2-6b-int4 llm_weights/chatglm2-6b-int4
fi
if [ ! -d "llm_weights/Qwen-7B-Chat-Int4" ]
then
    echo "Downloading..."
    git lfs clone https://huggingface.co/Qwen/Qwen-7B-Chat-Int4 llm_weights/Qwen-7B-Chat-Int4
fi
echo "Done."

