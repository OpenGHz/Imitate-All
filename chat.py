import base64
import requests
import cv2

# OpenAI API Key
# api_key = "sk-proj-gNS4ar7fAWypmYNjOpLmT3BlbkFJ5Nm2hX5Jx9iV4Jqkn85Q"
api_key = "sk-proj-ThHbAqYhOdLCTbYWfZnnT3BlbkFJ3sk2kwtniRqC4lvwzM0x"


# Read text from .txt file
def read_text_from_file(file_path):
  with open(file_path, "r") as file:
    text = file.read()
    return text

initial_prompt_path = "/home/dyh/OCIA/initial_prompt.txt"

initial_prompt = read_text_from_file(initial_prompt_path)

pass_me_the_bowl_path = '/home/dyh/OCIA/pass_me_the_fruit.txt'

pass_me_the_bowl = read_text_from_file(pass_me_the_bowl_path)

def capture_and_encode_image(camera_index):
    # Open USB camera
    cap = cv2.VideoCapture(camera_index)
    # Read image from camera
    ret, frame = cap.read()
    # Release camera
    cap.release()
    if not ret:
       raise ValueError("Failedto capture image")
    # Convert image to JPEG format
    _, image_buffer = cv2.imencode('.jpg', frame)
    # Encode image buffer to base64
    base64_image = base64.b64encode(image_buffer).decode('utf-8')
    return base64_image

# Set USB camera index
camera_index = 2  

# Capture and encode image from USB camera
base64_image = capture_and_encode_image(camera_index)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": initial_prompt
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        },
        {
          "type": "text",
          "text": pass_me_the_bowl
        }
      ]
    }
  ],
  "max_tokens": 720
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())