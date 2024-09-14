import os

import base64
from PIL import Image
from io import BytesIO

# Set up proxy if needed
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# Initialize the OpenAI client
from openai import OpenAI
client = OpenAI(api_key="")

def call_gpt4_api(image_path):
    try:
        # Read the image and encode it as base64
        with open(image_path, 'rb') as file:
            image_buffer = file.read()
        base64_image = base64.b64encode(image_buffer).decode('utf-8')

        # Construct the prompt
        prompt = (
            "Analyze the image and return a dictionary with 'status': "
            "0 if the object is dropped and not caught, "
            "1 if there is no object to catch, "
            "2 if there is too much clutter in the image."
        )

        # Call the OpenAI API
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}
            ],
            max_tokens=720
        )

        # Check the response
        if completion and completion.choices:
            return completion.choices[0].message.content
        else:
            print("Failed to get a valid response")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Call the function and print the result
result = call_gpt4_api('/home/qiuzhi/bowls.png')
if result:
    print(result)
else:
    print("Failed to get a valid response")