# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""OpenAI only. The key comes from .env (OPENAI_API_KEY)."""
import os, time

def call(model_cfg, system, user):
    from openai import OpenAI
    t0 = time.time(); cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try: r = cl.chat.completions.create(model=model_cfg["model_id"], messages=msgs, temperature=0); temp = 0
    except Exception as e:
        if "temperature" in str(e).lower(): r = cl.chat.completions.create(model=model_cfg["model_id"], messages=msgs); temp = "default"
        else: raise
    return dict(text=r.choices[0].message.content, in_tok=r.usage.prompt_tokens, out_tok=r.usage.completion_tokens, latency=time.time() - t0, temp=temp)
