from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

# Create a streaming chat completion
stream = client.chat.completions.create(
    model="smol-lm",  # Model name doesn't matter
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    stream=True,
)

# Print the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print()  # New line at the end