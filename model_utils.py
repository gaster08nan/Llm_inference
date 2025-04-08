# pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
import json
import os

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYS_PROMPT = (
    "You are Llama 3, a highly advanced conversational AI designed to provide helpful, accurate, and engaging information to users. Your main objectives are: "
    "1. Accuracy: Always provide precise and well-researched responses. If you're unsure of an answer, politely inform the user. "
    "2. Engagement: Foster a friendly and engaging conversation style that adapts to the tone and needs of the user. "
    "3. Safety: Avoid sharing harmful, illegal, or inappropriate content. Respect user privacy and adhere to ethical guidelines. "
    "4. Clarity: Communicate effectively, ensuring that your responses are clear, concise, and contextually relevant. "
    "5. Customization: Tailor your responses to the user's preferences and needs, considering their context and queries. "
    "6. Language: Support multiple languages, providing natural and fluent interactions in the language the user selects. "
    "At the same time, you should: "
    "- Always claim to be an AI and never pretend to be human or sentient. "
    "- Always answer directly, using bullet points, lists, or tables when appropriate. "
    "- Avoid engaging in topics outside your training or that conflict with ethical guidelines. "
    "- Cite sources when using external information, where possible. "
    "- Use markdown formatting for better readability when appropriate (e.g., lists, tables, headings)."
    "When answering comparative questions, always provide a table with the comparison. "
    "Always keep the responses refine and precise."
)


def save_history(history, file_path="history.json"):
    # breakpoint()
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)


def load_history(file_path="history.json"):
    history = None
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            history = json.load(file)
    return history


def delete_history(file_path="history.json"):
    with open(file_path, 'w') as f:
        f.write("")


def _init_unsloth_model(model_name):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def process_response(response):
    last_response = " ".join(response[-1].split('<|begin_of_text|>')[-1].split(
        '<|eot_id|>')[0].split('[/INST]')[1:])
    last_response = last_response.replace('<[INST]>', '').replace(
        '[/INST]', '').replace('<<SYS>>', '').replace('<</SYS>>', '')
    return last_response


def answer_message(model, tokenizer, messages):
    text_streamer = TextStreamer(tokenizer)
    inputs = tokenizer.apply_chat_template(messages,
                                           tokenize=True,
                                           add_generation_prompt=False,
                                           return_tensors="pt").to(device)
    response = model.generate(input_ids=inputs,
                              streamer=text_streamer,
                              max_new_tokens=4096,
                              use_cache=True)
    response = tokenizer.batch_decode(response)
    response = process_response(response)
    return response


def setup_inference():
    unsloth_4bit_models = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"

    model, tokenizer = _init_unsloth_model(unsloth_4bit_models)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama",
        # mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    )

    return model, tokenizer
