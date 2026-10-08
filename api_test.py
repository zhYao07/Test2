import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("generate_label"),
    base_url="https://api.catbeeai.com/v1",
)

response = client.responses.create(
    model="gpt-6.1-sol",
    input="你是什么模型，具体什么型号"
)

print(response.output_text)