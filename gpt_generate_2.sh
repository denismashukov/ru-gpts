#!/bin/bash
python3 ru-gpts/generate_transformers_2.py \
    --model_type=gpt2
    --model_name_or_path=models/feedbacks_large_2_2048  \
    --k=50    \
    --p=0.95    \
    --length=128   \
    --repetition_penalty=5 \
    --no_repeat_ngram_size=3 \
    --num_return_sequences=1 \
    --seed=4200 \
    --temperature=0.35 \
    --num_beams=10 \

# chmod +x ru-gpts/gpt_generate_2.sh
# nohup ru-gpts/gpt_generate_2.sh &

# parse feedbacks_seeds.csv generated by generate_transformers_1.py
# take first sentence and generate with more strict temperature