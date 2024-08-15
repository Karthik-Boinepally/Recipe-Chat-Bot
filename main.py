import json
from sentence_transformers import SentenceTransformer
from htmlChunking import parse_html
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llmModel import get_gpt_answer_with_retry

def load_json(json_file):
    with open(json_file, 'r') as file:
        content_dict = json.load(file)
    return content_dict

def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)

def check_embedding_shapes(embeddings):
    for key, value in embeddings.items():
        print(f"Key: {key}, Shape: {value.shape}")

def average_embeddings(embeddings):
    for key in embeddings:
        if len(embeddings[key].shape) > 1:
            embeddings[key] = np.mean(embeddings[key], axis=0)
    return embeddings


def json_to_embeddings(json_content):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = {}

    for heading, text_list in json_content.items():
        # Combine the heading with its content
        combined_text = heading + " " + " ".join(text_list)

        # Generate embedding for each chunk of content
        embedding = model.encode(combined_text)
        embeddings[heading] = embedding

    return embeddings

def load_embeddings(filename):
    return np.load(filename, allow_pickle=True).item()

def get_top_n_similar_embeddings(user_embedding, content_embeddings, top_n=3):
    keys = list(content_embeddings.keys())
    vectors = list(content_embeddings.values())
    similarities = cosine_similarity([user_embedding], vectors)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_matches = [(keys[i], similarities[i]) for i in top_indices]
    return top_matches

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    url = input("Please enter the url").strip()
    parsed_content = parse_html(url)
    print(parsed_content)
    json_file = 'output.json'

    content_dict = load_json(json_file)
    embeddings = json_to_embeddings(content_dict)
    save_embeddings(embeddings, 'content_embeddings.npy')
    print("Embeddings have been saved to 'content_embeddings.npy'.")

    content_embeddings = load_embeddings('content_embeddings.npy')
    check_embedding_shapes(content_embeddings)
    content_embeddings = average_embeddings(content_embeddings)

    while True:
        user_question = input("Please enter your question (or type 'exit' to quit): ").strip()

        if user_question.lower() == 'exit':
            print("Goodbye!")
            break

        user_embedding = model.encode(user_question)
        top_matches = get_top_n_similar_embeddings(user_embedding, content_embeddings, top_n=3)

        # print("Top 3 matches:")
        # for heading, similarity in top_matches:
        #     print(f"Heading: {heading}, Similarity: {similarity:.4f}")

        top_heading, top_similarity = top_matches[0]
        top_content = content_dict.get(top_heading, "")

        if not top_content:
            print(f"Error: No content found for heading '{top_heading}'.")
            continue

        # print(f"Top match: {top_heading}, Similarity: {top_similarity:.4f}")
        # print(f"Top Content: {top_content}\n")

        answer = get_gpt_answer_with_retry(user_question, content_dict)
        print(f"Answer: {answer}\n")

if __name__ == '__main__':
    main()
