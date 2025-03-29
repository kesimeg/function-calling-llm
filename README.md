# function-calling-llm

## Overview

In this repo we finetune LLama-3.1-8B model for function calling using [Unsloth](https://github.com/unslothai/unsloth).

For function calling [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset is used. As well as [Alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset to keep instruction following feature.

We select a subset of the original datasets using a reward model. In the dataset preparation notebook you will see that some samples in the function calling dataset is inappropriate. Also some answers in the Alpaca dataset are irrelevant to question.
You can find the trained model on [Huggingface](https://huggingface.co/kesimeg/function-calling-llama-3.1-8B)


## How to use the Model

Use the code below to get started with the model.
```python
from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
from unsloth.chat_templates import get_chat_template
max_seq_length = 4096


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "kesimeg/function-calling-llama-3.1-8B",
    max_seq_length = max_seq_length,
)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

FastLanguageModel.for_inference(model)

#For instruction prompts use the code below
instruction = "What is the integral of cos(x)"
convos = [{"role":"user","content":instruction}]
texts = tokenizer.apply_chat_template(convos,tokenize = False, add_generation_prompt = True)

inputs = tokenizer(
[    texts
], return_tensors = "pt").to("cuda")


text_streamer = TextStreamer(tokenizer, skip_prompt = False)
outputs = model.generate(**inputs, max_new_tokens = 4096, streamer = text_streamer)


#For function calling use the following code
query = """Make an approprite function call according to user query:I'm trying to get a
 specific number of products from the catalog, let's say 15, but I don't want
 to start from the beginning. I want to skip the first 200 products. Can you
 help me with that?"""
 
tool_object = """[{"name": "get_products", "description":\
 "Fetches a list of products from an external API with optional query\
 parameters for limiting and skipping items in the response.", "parameters":\
 {"limit": {"description": "The number of products to return.", "type":\
 "int", "default": ""}, "skip": {"description": "The number of products to\
 skip in the response.", "type": "int", "default": ""}}}]"""

texts = tokenizer.apply_chat_template(convos,tools=tool_object,tokenize = False, add_generation_prompt = False)
texts = texts.replace('"parameters": d','"arguments": d') # original tool use function uses parameters our dataset uses arguments

convos = [{"role":"user","content":texts}]
texts = tokenizer.apply_chat_template(convos,tokenize = False, add_generation_prompt = True)

inputs = tokenizer(
[    texts
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
outputs = model.generate(**inputs, max_new_tokens = 4096, streamer = text_streamer)
```
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
