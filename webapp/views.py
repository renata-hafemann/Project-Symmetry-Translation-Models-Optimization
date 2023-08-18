from django.shortcuts import render
from .scraping_utils import scrape_article
from .translation_utils import translate_with_googletrans, translate_with_marian, translate_with_t5
from transformers import BertTokenizer, BertModel
import torch
from django.http import JsonResponse, HttpResponseBadRequest
import time


def BASE(request):
    return render(request, 'base.html')

def scrape(request):
    if request.method == 'POST':
        article_input = request.POST.get('article_input')
        language = request.POST.get('language', 'en')  # Set the language of your choice, default: English

        # Limit the length of the article_input to fit within the allowed limit
        article_input = article_input[:300]

        start_time = time.time()  # Start measuring execution time for scraping
        scraped_content = scrape_article(article_input, language)
        execution_time = calculate_execution_time(start_time)  # Calculate execution time for scraping

        if scraped_content:
            word_count = len(scraped_content.split())
            paragraph_count = scraped_content.count('\n\n') + 1
            article_length = len(scraped_content)

            response_data = {
                'article': article_input,
                'content': scraped_content,
                'word_count': word_count,
                'paragraph_count': paragraph_count,
                'length_article': article_length,
                'execution_time': execution_time
            }

            return JsonResponse(response_data)
        else:
            error_message = f"Wikipedia page not found for article: {article_input}"
            print(error_message)
            return JsonResponse({'error': error_message}, status=404)


def calculate_execution_time(start_time):
    end_time = time.time()
    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def translate(request):
    if request.method == 'POST':
        source_article = request.POST.get('source_article')
        source_language = request.POST.get('source_language')  
        destination_language = request.POST.get('destination_language')
        translation_model = request.POST.get('translation_model')

        start_time = time.time()  # Start measuring execution time for translation

        if translation_model == 'googletranslate':
            translated_text = translate_with_googletrans(source_article, destination_language)
        elif translation_model == 'Marian':
            translated_text = translate_with_marian(source_article, source_language, destination_language)
        elif translation_model == 'T5':
            translated_text = translate_with_t5(source_article, source_language, destination_language)


        execution_time = calculate_execution_time(start_time)  # Calculate execution time for translation

        word_count = len(translated_text.split()) if translated_text else 0
        paragraph_count = translated_text.count('\n\n') + 1 if translated_text else 0
        length_article = len(translated_text) if translated_text else 0

        translated_statistics = {
            'source_article': source_article,
            'translated_text': translated_text,
            'word_count': word_count,
            'paragraph_count': paragraph_count,
            'length_article': length_article,
            'execution_time': execution_time
        }

        return JsonResponse(translated_statistics)
    
def highlight_similar_sentences(translated_text, destination_text, batch_size=64):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    # Check if translated_text and destination_text are None
    if translated_text is None or destination_text is None:
        return {
            'highlighted_translated_article': "Article not found.",
            'highlighted_destination_article': "Article not found."
        }

    # Convert the translated_text and destination_text tensors to a list of sentences if they are not already
    if not isinstance(translated_text, list):
        translated_text = translated_text.split('. ')
    if not isinstance(destination_text, list):
        destination_text = destination_text.split('. ')

    # Pad or truncate the sentences to match the length of the longest article
    max_num_sentences = max(len(translated_text), len(destination_text))
    translated_text = translated_text + [''] * (max_num_sentences - len(translated_text))
    destination_text = destination_text + [''] * (max_num_sentences - len(destination_text))

    # Split the sentences into batches
    num_batches = (max_num_sentences + batch_size - 1) // batch_size
    batched_translated_text = [translated_text[i:i + batch_size] for i in range(0, max_num_sentences, batch_size)]
    batched_destination_text = [destination_text[i:i + batch_size] for i in range(0, max_num_sentences, batch_size)]

    # Initialize lists to store the highlighted sentences for each batch
    highlighted_translated_sentences_list = []
    highlighted_destination_sentences_list = []

    # Process each batch separately
    for batch_translated_text, batch_destination_text in zip(batched_translated_text, batched_destination_text):
        # Encode the sentences with language information as a list of strings
        inputs = tokenizer(batch_translated_text + batch_destination_text, return_tensors='pt', padding=True, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        inputs['attention_mask'] = inputs['attention_mask'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        with torch.no_grad():
            outputs = model(**inputs)

        # Calculate embeddings for all sentences
        sentence_embeddings = torch.mean(outputs.last_hidden_state, dim=1)

        # Compute cosine similarity between each translated and destination sentence
        similarity_scores = torch.nn.functional.cosine_similarity(sentence_embeddings[:len(batch_translated_text)], sentence_embeddings[len(batch_translated_text):], dim=1)

        # Highlight sentences with similarity >= threshold (you can adjust this threshold)
        threshold = 0.6

        # Highlight the translated sentences for this batch
        highlighted_translated_sentences = [
            f"<span class='highlight-green'>{sentence}</span>" if any(score >= threshold for score in similarity_scores) else f"<span class='highlight-red'>{sentence}</span>"
            for sentence in batch_translated_text
        ]
        highlighted_translated_sentences_list.extend(highlighted_translated_sentences)

        # Highlight the destination sentences for this batch
        highlighted_destination_sentences = [
            f"<span class='highlight-green'>{sentence}</span>" if score >= threshold else f"<span class='highlight-red'>{sentence}</span>"
            for sentence, score in zip(batch_destination_text, similarity_scores)
        ]
        highlighted_destination_sentences_list.extend(highlighted_destination_sentences)

    # Join the highlighted sentences from all batches
    translated_text_highlighted = '. '.join(highlighted_translated_sentences_list)
    destination_text_highlighted = '. '.join(highlighted_destination_sentences_list)

    # Return the highlighted articles as a dictionary
    return {
        'highlighted_translated_article': translated_text_highlighted,
        'highlighted_destination_article': destination_text_highlighted,
    }
    
def compare_sentences(request):
    if request.method == 'POST':
        translated_article = request.POST.get('translated_article')
        destination_article = request.POST.get('destination_article')

        # Check if both articles are provided
        if translated_article is None or destination_article is None:
            return JsonResponse({
                'error': "Both translated_article and destination_article are required."
            }, status=400)

        # Perform sentence similarity comparison and highlighting
        highlighted_articles = highlight_similar_sentences(translated_article, destination_article)

        # Calculate statistics for translated_article and destination_article
        translated_word_count = len(translated_article.split())
        translated_paragraph_count = translated_article.count('\n\n') + 1
        translated_length_article = len(translated_article)
        translated_execution_time = "00:00:00"  # Execution time is not applicable for this endpoint

        source_word_count = len(destination_article.split())
        source_paragraph_count = destination_article.count('\n\n') + 1
        source_length_article = len(destination_article)
        source_execution_time = "00:00:00"  # Execution time is not applicable for this endpoint

        # Update the statistics in the highlighted_articles object
        highlighted_articles['translated_word_count'] = translated_word_count
        highlighted_articles['translated_paragraph_count'] = translated_paragraph_count
        highlighted_articles['translated_length_article'] = translated_length_article
        highlighted_articles['translated_execution_time'] = translated_execution_time
        highlighted_articles['source_word_count'] = source_word_count
        highlighted_articles['source_paragraph_count'] = source_paragraph_count
        highlighted_articles['source_length_article'] = source_length_article
        highlighted_articles['source_execution_time'] = source_execution_time

        # Additional statistics
        similarity_threshold = 0.6  # You can adjust this threshold
        similar_sentences_count = highlighted_articles['highlighted_translated_article'].count('highlight-green')
        different_sentences_count = highlighted_articles['highlighted_translated_article'].count('highlight-red')

        highlighted_articles['similar_sentences_count'] = similar_sentences_count
        highlighted_articles['different_sentences_count'] = different_sentences_count

        return JsonResponse(highlighted_articles)
    else:
        return HttpResponseBadRequest("Invalid request method. Only POST is allowed.")
    