import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

def load_tokenizer_and_model(model_name_or_path):
    return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()

def generate(
        model, tok, text,
        do_sample=True, max_length=100, repetition_penalty=5.0,
        top_k=5, top_p=0.95, temperature=1.0,
        num_beams=None,
        no_repeat_ngram_size=3,
        early_stopping=True
):
    input_ids = tok.encode(text, return_tensors="pt").cuda()
    out = model.generate(
        input_ids.cuda(),
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k, top_p=top_p, temperature=temperature,
        num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping
    )
    return list(map(tok.decode, out))

tok, model = load_tokenizer_and_model("model_hf")

import tensorflow as tf
tf.random.set_seed(0)

text = []
for index in range(0, 200):
    generated = generate(model, tok, "feedback: ", num_beams=5, top_p=0.95, top_k=50,  temperature=1.0)
    print("Generated sequence: ", index)
    text.append(generated)

df = pd.DataFrame({'feedback': text})
df.to_csv('/home/ubuntu/feedbacks.csv', index=False)

