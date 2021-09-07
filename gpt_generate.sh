#!/bin/bash
python3 ru-gpts/generate_transformers.py \
    --model_type=gpt2 \
    --model_name_or_path=models/feedbacks_large_2_2048 \
    --k=50 \
    --p=0.95 \
    --length=256 \
    --num_beams=5 \
    --temperature=1 \
    --repetition_penalty=5 \
    --no_repeat_ngram_size=3

# chmod +x ru-gpts/gpt_generate.sh
# nohup ru-gpts/gpt_generate.sh &
