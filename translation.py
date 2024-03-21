from langdetect import detect
import spacy
import time
import torch

device = torch.device("cuda")


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

def translate(text, translation_model, tokenizer, spacy_model, extract_pos=False):
    # Split input text into smaller bits to be able to translate longer texts 
    doc = spacy_model(text)
    
    translations = []
    for s in doc.sents:
        if len(s) == 1:
            continue

        # Tokenize the source text
        inputs = tokenizer.encode(str(s), return_tensors="pt")

        # Perform the translation and decode the output
        outputs = translation_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0])
        translations.append(translated_text) 

    full_translation = "\n".join(translations)
    if extract_pos:
        noun_pairs, verb_pairs = translate_pos(doc, model=translation_model, tokenizer=tokenizer)
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