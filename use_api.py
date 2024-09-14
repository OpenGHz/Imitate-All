import requests
import cv2
import base64
import re
import time

# Read text from .txt file
def read_text_from_file(file_path):
  with open(file_path, "r") as file:
    text = file.read()
    return text

# Capture the image from the camera
def capture_image(camera_index):
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Failed to capture image")
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

# Call gpt4 api for plans and instructions
def call_gpt4_for_planning(initial_prompt = None, image_base64 = None, instruction = None):
    api_key = ""
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "system",
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": initial_prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
              }
            },
            {
              "type": "text",
              "text": instruction
            }
          ]
        }
      ]
    }
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

# Call gpt4 api for plans and instructions
def call_gpt4_for_judgement(prompt = None, image_base64 = None):
    api_key = "sk-proj-ThHbAqYhOdLCTbYWfZnnT3BlbkFJ3sk2kwtniRqC4lvwzM0x"
    api_url = "https://api.openai.com/v1/chat/completions"
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
              "text": prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
              }
            }
          ]
        }
      ]
    }
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

# Simulate actions of the robot arm
def delay(seconds, step_command):
    print(f"Executing:{step_command}")
    time.sleep(seconds)
    print("Delay completed")

# Extract steps
def extract_steps(text):
    lines = text.split('\n')
    steps = []
    step_pattern = re.compile(r'^\d+\..*')

    for line in lines:
        if step_pattern.match(line.strip()):
            steps.append(line.strip())
 
    return steps

# Extract objects
def extract_objects(text):
    objects_part = re.search(r"Objects:\s*", text)
    if not objects_part:
        return "No objects section found."
    
    objects_text = text[objects_part.end():]
    object_lists = re.findall(r"\[\d+\](.*?)(?=\n\[\d+\]|$)", objects_text)
    objects = [obj.strip().split(',') for obj in object_lists if obj.strip()]

    return objects

def main():
    initial_prompt_path = "/home/dyh/OCIA/initial_prompt.txt"
    initial_prompt = read_text_from_file(initial_prompt_path)
    instruction_path = "/home/dyh/OCIA/pass_me_the_fruit.txt"
    instrucntion = read_text_from_file(instruction_path)
    steps = []  # This will be filled after the initial API call
    objects = []
    camera_index = 2

    # Initial call to get the plan
    initial_image_base64 = capture_image(camera_index)
    plan_response = call_gpt4_for_planning(initial_prompt, initial_image_base64, instrucntion)
    print( plan_response)
    plan_text = plan_response['choices'][0]['message']['content']
    steps = extract_steps(plan_text)
    objects = extract_objects(plan_text)

    # Execute the steps according to the plan
    for step in steps:
        print(step)
        if "Done" in step:
            print("Task completed successfully.")
            break

        while True:
            delay(5, step_command = step)
            current_image_base64 = capture_image(camera_index)
            step_prompt = f"Here is the image after executing {step}. Is it done correctly? The format of your answer should only be one word: correct or false."
            verification_response = call_gpt4_for_judgement(prompt = step_prompt, image_base64 = current_image_base64)
            print(verification_response)
            verification_text = verification_response['choices'][0]['message']['content']
            print(verification_text)
            if "correct" in verification_text:
                print("Step completed correctly.")
                break
            if "Correct" in verification_text:
                print("Step completed correctly.")
                break
            else:
                print("Step not completed correctly, retrying...")

if __name__ == "__main__":
    main()