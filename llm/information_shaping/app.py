import gradio as gr
from chat_on_docs import ChatDoc

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    agent = ChatDoc()

    def respond(message, chat_history):
        bot_message = agent.chat(message)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()