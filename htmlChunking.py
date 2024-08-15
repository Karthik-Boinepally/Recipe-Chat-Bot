import requests
from bs4 import BeautifulSoup
import json
from transformers import GPT2Tokenizer

# Initialize the tokenizer (you can use other models like BERT as well)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def parse_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Ensure we got a valid response
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')

    content_dict = {}

    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        heading_text = heading.get_text(strip=True)
        content = []
        for sibling in heading.find_next_siblings():
            if sibling.name and sibling.name.startswith('h'):
                break
            content.append(sibling.get_text(strip=True))

        # Combine the content list into a single string
        content_text = " ".join(content)

        # Tokenize and chunk the content if it exceeds 256 tokens
        tokens = tokenizer.encode(content_text)
        chunks = []
        for i in range(0, len(tokens), 1000):
            chunk = tokens[i:i + 1000]
            chunk_text = tokenizer.decode(chunk)
            chunks.append(chunk_text)

        # Store the chunks in the dictionary under the heading
        content_dict[heading_text] = chunks

    with open('output.json', 'w') as json_file:
        json.dump(content_dict, json_file, indent=4)
    # Convert the dictionary to a JSON formatted string
    content_json = json.dumps(content_dict, indent=4)

    # json_to_embeddings(content_json)

    return content_json


# Example usage
# url = 'https://www.bonappetit.com/recipe/chicken-piccata-2'
# parsed_content = parse_html(url)

