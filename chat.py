from PIL import Image
from langdetect import detect
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr
import os
import pytesseract
import spacy
import time
import torch


device = torch.device("cuda")

# Load models 
it_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
it_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-en")

en_it_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
en_it_model =  AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")

"Helsinki-NLP/opus-mt-en-it"
it_nlp = spacy.load("it_core_news_sm")
en_nlp = spacy.load("en_core_web_sm")


def language_detection(text:str)->str:    
    """
    @param text: text to check
    @return: language iso code. Full list here https://github.com/Mimino666/langdetect#languages
    """
    if not text or not text.strip():
        return "unknown"
    return detect(text)

def translate_pos(spacy_doc, model, tokenizer):
    nouns = [token.text for token in spacy_doc if token.pos_ == "NOUN"]
    verbs = [token.text for token in spacy_doc if token.pos_ == "VERB"]

    # Translate each unique noun and verb
    unique_nouns = list(set(nouns))
    unique_verbs = list(set(verbs))
    noun_translations = translate_words(unique_nouns, model, tokenizer)
    verb_translations = translate_words(unique_verbs, model, tokenizer)

    # Create dictionaries mapping the original words to their translations
    noun_pairs = dict(zip(unique_nouns, noun_translations))
    verb_pairs = dict(zip(unique_verbs, verb_translations))

    # Replace the original words with their translations
    translated_nouns = [noun_pairs[noun] for noun in nouns]
    translated_verbs = [verb_pairs[verb] for verb in verbs]

    return dict(zip(nouns, translated_nouns)), dict(zip(verbs, translated_verbs))

def translate(text, model, tokenizer, extract_pos=False):
    # Split input text into smaller bits to be able to translate longer texts 
    doc = it_nlp(text)
    
    translations = []
    for s in doc.sents:
        if len(s) == 1:
            continue

        # Tokenize the source text
        inputs = tokenizer.encode(str(s), return_tensors="pt")

        # Perform the translation and decode the output
        outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0])
        translations.append(translated_text) 

    full_translation = "\n".join(translations)
    if extract_pos:
        noun_pairs, verb_pairs = translate_pos(doc)
        pos_output = "\nNouns:\n" + "\n".join([f"| {noun} -> {translation}" for noun, translation in noun_pairs.items()])
        pos_output += "\n\nVerbs:\n" + "\n".join([f"| {verb} -> {translation}" for verb, translation in verb_pairs.items()])
        full_translation += "\n\n" + pos_output

    return full_translation.replace("<pad>", "").replace("</s>", "")

def translate_words(words, model, tokenizer):
    # Tokenize the source words
    inputs = tokenizer.batch_encode_plus(words, return_tensors="pt", padding=True)

    # Perform the translation and decode the output
    outputs = model.generate(**inputs, max_length=40, num_beams=4, early_stopping=True)
    translated_words = [tokenizer.decode(output) for output in outputs]

    return translated_words


def ocr_from_image(image_path):
    # Open the image file
    img = Image.open(image_path)
    # Use pytesseract to do OCR on the image
    text = pytesseract.image_to_string(img)
    return text

# Use the function
image_path = 'path_to_your_image.png'
print(ocr_from_image(image_path))



def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


def bot(history):
    user_message = history[-1][0]

    lang = language_detection(user_message)

    if lang == "en":
        translated_message = translate(user_message, en_it_model, en_it_tokenizer)

    elif lang == "it": 
        translated_message = translate(user_message, it_en_model, it_en_tokenizer)
    
    else: 
        translated_message = translate(user_message, it_en_model, it_en_tokenizer)

    history[-1][1] = translated_message
    yield history



with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
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