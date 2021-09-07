#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 ru-gpts/pretrain_transformers.py    \
--output_dir=models/feedbacks_large    \
--model_type=gpt2    \
--model_name_or_path=sberbank-ai/rugpt3large_based_on_gpt2     \
--do_train    \
--train_data_file=train.txt    \
--do_eval  \
--eval_data_file=valid.txt  \
--per_gpu_train_batch_size=1   \
--gradient_accumulation_steps=1  \
--num_train_epochs=3   \
--block_size=256    \
--overwrite_output_dir \
 --save_steps 1000000
