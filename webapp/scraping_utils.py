import wikipedia


def scrape_article(article_input, language='en'):
    try:
        wikipedia.set_lang(language)
        article_title = extract_article_title(article_input)
        page = wikipedia.WikipediaPage(title=article_title)
        scraped_content = page.content
    except wikipedia.exceptions.PageError:
        return None

    return scraped_content

def extract_article_title(article_input):
    # Extract the article title from the link
    if '/wiki/' in article_input:
        # Extract the part after '/wiki/' to get the title
        article_title = article_input.split('/wiki/')[1]
        # Remove any fragments in the URL after the title
        if '#' in article_title:
            article_title = article_title.split('#')[0]
        return article_title.replace('_', ' ')  # Replace underscores with spaces in the title
    else:
        # If the link doesn't follow the '/wiki/' format, return the user's input directly
        return article_input
    
    
