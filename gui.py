import time

import gradio as gr
from gradio import ChatMessage

import model_utils as utils

system_prompt = utils.SYS_PROMPT

model, tokenizer = utils.setup_inference()
history = [
    {
        "role": "system",
        "content": system_prompt
    },
]
history_file = "history.json"


def add_message(message, history):
    history.append(ChatMessage(role="user", content=message['text']))
    return "", history


def response_message(history):
    start_time = time.time()
    response = utils.answer_message(model, tokenizer, history)
    ans_time = time.time() - start_time
    history.append(ChatMessage(role="assistant", content=response))

    return history, f"Last answer response time: {ans_time:.3f}s"


def save_history(history):
    utils.save_history(history, history_file)


def delete_history():
    utils.delete_history(history_file)
    return ""


with gr.Blocks() as demo:
    history = utils.load_history(history_file) or history
    chatbot = gr.Chatbot(
        value=history,
        type="messages",
        height=800,
        max_height=900,
        min_height=600,
    )
    ans_time_output = gr.Markdown("Last answer response time: 0s")
    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
        sources=["microphone", "upload"],
    )
    clear = gr.Button("Clear History")
    chat_msg = chat_input.submit(add_message, [chat_input, chatbot],
                                 [chat_input, chatbot])
    chat_msg = chat_msg.then(response_message,
                             inputs=chatbot,
                             outputs=[chatbot, ans_time_output])
    chat_msg.then(save_history, chatbot, None)
    clear.click(delete_history, None, chatbot)

if __name__ == "__main__":
    demo.launch(share=True)
