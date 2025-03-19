import time
import gradio as gr
import model_utils as utils
from gradio import ChatMessage

system_prompt = utils.SYS_PROMPT

model, tokenizer = utils.setup_inference()
history =  [
        {"role": "system", "content": system_prompt},
    ]


def response_message(message, history):
    breakpoint()
    history.append({"role":"user", "content": message})
    # history.append(ChatMessage(role="user", content=message))
    response = utils.answer_message(model, tokenizer, history)
    history = history.append({"role":"assistant", "content": response})
    # history.append(ChatMessage(role="assistant", content=response))
    time.sleep(1)
    return response, history

def bot_responser(history):
    output_response = ''
    if history['-1']['role'] == 'assistant':
        for character in history['-1']['content']:
            output_response += character
            time.sleep(0.3)
            yield output_response

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
        sources=["microphone", "upload"],
    )
    print(chat_input)
    chat_msg = chat_input.submit(response_message, [chat_input, chatbot], [chat_input, chatbot])
    bot_msg = chat_msg.then(bot_responser,  chatbot, chatbot, api_name="bot_response") #yeild the response message
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]) # clear the chat input bar


demo.launch(share=True)
