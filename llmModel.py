import time
import openai
from openai.error import RateLimitError
# Set your OpenAI API key
# openai.api_key = "Contact me for API Key or use yours"
def get_gpt_answer_with_retry(question, context, retries=5):
    prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
    for i in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # or "gpt-3.5-turbo"
                messages=[
                    {"role": "system", "content": "You are an assistant to answer questions about recipes. Answer only from the given context. They are usually steps and ingredients"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response['choices'][0]['message']['content'].strip()
        except RateLimitError:
            if i < retries - 1:
                wait_time = (2 ** i) * 5
                print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
