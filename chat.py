import os
import gradio as gr
import spacy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from translation import translate, language_detection
from ocr import ocr_from_image

# Load models 
it_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
it_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-en")

en_it_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
en_it_model =  AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")

it_nlp = spacy.load("it_core_news_sm")
en_nlp = spacy.load("en_core_web_sm")

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)

def add_file(history, file):
    # Save the uploaded file
    file_path = os.path.join(os.getcwd(), file.name)
    print(file_path)
    # with open(file_path, 'wb') as f:
    #     f.write(file)
    # Extract text from the image
    extracted_text = ocr_from_image(file_path)
    # Check if the extracted text is not empty
    if extracted_text.strip():
        # Add the extracted text to the chat history
        history = history + [(extracted_text, None)]
    return history

def bot(history):
    user_message = history[-1][0]
    lang = language_detection(user_message)

    if lang == "en":
        translated_message = translate(user_message, translation_model=en_it_model, tokenizer=en_it_tokenizer, spacy_model=en_nlp, enextract_pos=False)
    elif lang == "it": 
        translated_message = translate(user_message, translation_model=it_en_model, tokenizer=it_en_tokenizer, spacy_model=it_nlp, extract_pos=True)
    else: 
        translated_message = translate(user_message, translation_model=it_en_model, tokenizer=it_en_tokenizer, spacy_model=it_nlp, extract_pos=True)

    history[-1][1] = translated_message
    yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=True,
        layout="bubble",
        height=500,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    chatbot.like(print_like_dislike, None, None)

demo.queue()
if __name__ == "__main__":
    demo.launch()
