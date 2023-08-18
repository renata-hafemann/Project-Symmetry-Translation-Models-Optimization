from googletrans import Translator
from transformers import MarianMTModel, MarianTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from nltk.tokenize import sent_tokenize



def translate_with_googletrans(text, target_language):
    translated_content = ""

    translator = Translator()
    for section in text.split("\n\n"):
        try:
            translated_section = translator.translate(section, dest=target_language)
            if translated_section.text:
                translated_content += translated_section.text + "\n\n"
        except Exception as e:
            print(f"An error occurred during translation: {str(e)}")

    return translated_content

def translate_with_marian(text, source_language, target_language, max_length=1000):
    if not source_language or not target_language:
        # If source_language or target_language is not provided, return an error message
        return "Error: Source and target languages must be specified."

    model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language}'
    token = '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'  # Replace this with the actual token you generated

    tokenizer = MarianTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = MarianMTModel.from_pretrained(model_name, use_auth_token=token)

    # Split the input text into chunks
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]

    # Translate each chunk separately
    translated_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        translated = model.generate(**inputs, max_new_tokens=max_length)
        translated_chunk = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_chunks.append(translated_chunk)

    # Join the translated chunks to get the final translated text
    translated_text = ' '.join(translated_chunks)

    return translated_text

# Load the T5 model and tokenizer
translation_pipeline_fr = pipeline("translation_en_to_fr", model="t5-small", tokenizer="t5-small")
translation_pipeline_de = pipeline("translation_en_to_de", model="t5-small", tokenizer="t5-small")
translation_pipeline_fr_to_de = pipeline("translation_fr_to_de", model="t5-small", tokenizer="t5-small")
translation_pipeline_fr_to_en = pipeline("translation_fr_to_en", model="t5-small", tokenizer="t5-small")
translation_pipeline_de_fr = pipeline("translation_de_to_fr", model="t5-small", tokenizer="t5-small")
translation_pipeline_de_en = pipeline("translation_de_to_en", model="t5-small", tokenizer="t5-small")
translation_pipeline_en_to_ro = pipeline("translation_en_to_ro", model="t5-small", tokenizer="t5-small")
translation_pipeline_ro_to_en = pipeline("translation_ro_to_en", model="t5-small", tokenizer="t5-small")

def translate_with_t5(source_article, source_language, destination_language):
    # Split the source article into paragraphs (lines)
    paragraphs = source_article.split('\n\n')

    translated_content = ''  # Initialize the variable to accumulate translated content
    
    for paragraph in paragraphs:
        translated_chunk = None
        
        # Translate each paragraph using T5 based on source and destination languages
        if source_language == 'en':
            if destination_language == 'fr':
                translated_chunk = translation_pipeline_fr(paragraph, max_length=512, truncation=True)[0]['translation_text']
            elif destination_language == 'de':
                translated_chunk = translation_pipeline_de(paragraph, max_length=512, truncation=True)[0]['translation_text']
            elif destination_language == 'ro':
                translated_chunk = translation_pipeline_en_to_ro(paragraph, max_length=512, truncation=True)[0]['translation_text']
        elif source_language == 'fr':
            if destination_language == 'de':
                translated_chunk = translation_pipeline_fr_to_de(paragraph, max_length=512, truncation=True)[0]['translation_text']
            elif destination_language == 'en':
                translated_chunk = translation_pipeline_fr_to_en(paragraph, max_length=512, truncation=True)[0]['translation_text']
        elif source_language == 'de':
            if destination_language == 'fr':
                translated_chunk = translation_pipeline_de_fr(paragraph, max_length=512, truncation=True)[0]['translation_text']
            elif destination_language == 'en':
                translated_chunk = translation_pipeline_de_en(paragraph, max_length=512, truncation=True)[0]['translation_text']
        elif source_language =='ro':
            if destination_language == 'en':
                translated_chunk = translation_pipeline_ro_to_en(paragraph, max_length=512, truncation=True)[0]['translation_text']
        
        if translated_chunk:
            translated_content += translated_chunk + '\n\n'  # Add double newline after each translated paragraph
    
    return translated_content  # Return the accumulated translated content after processing all paragraphs

