from functools import partial
import os
import PIL
import cv2
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image, ImageTk
import time
import tkinter as tk
import torch


#
# Event Handlers
#

def on_close():
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()


def loop(device):
    ret, frame = cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # downsample to 224x224
    input_frame = cv2.resize(rgb_frame, (224, 224))
    pil_image = Image.fromarray(input_frame)

    start_time = time.perf_counter()
    inputs = processor(prompt, pil_image, return_tensors="pt").to(device)
    print(f"input device {inputs['pixel_values'].device}")
    output = model.generate(**inputs, max_new_tokens=20)
    print(f"model device {output.device}")
    result = processor.decode(output[0], skip_special_tokens=True)[len(prompt):]
    result = result.replace("\n", " ").strip(" ")
    elapsed_time = time.perf_counter() - start_time
    print(f"Inference result '{result}' in {elapsed_time:.4f} seconds")

    # check if result contains keyword "hard hat" or "helmet"
    if "hard hat" in result or "helmet" in result:
        statusLabel.config(text="OK", fg="green")
    else:
        statusLabel.config(text="Not OK", fg="red")

    resized_frame = cv2.resize(rgb_frame, (window_width, window_height))
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(resized_frame))
    imageCanvas.create_image(0, 0, image=photo, anchor=tk.NW)
    imageCanvas.image = photo

    window.after(1000, partial(loop, device='cuda' if torch.cuda.is_available() else 'cpu'))


#
# User Interface
#

window_width = 800
window_height = 600
text_size = 80
text_color = "green"

cap = cv2.VideoCapture(0)

window = tk.Tk()
window.title("Webcam with OK Text")
window.geometry(f"{window_width}x{window_height+100}")

imageCanvas = tk.Canvas(window, width=window_width, height=window_height)
imageCanvas.pack(side=tk.BOTTOM)

statusLabel = tk.Label(
    window, text="OK", font=(None, text_size), fg=text_color, bg="white"
)
statusLabel.pack(side=tk.TOP)

window.protocol("WM_DELETE_WINDOW", on_close)

#
# Model
#

model_id = "google/paligemma-3b-mix-224"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

hg_token = os.environ.get("HUGGINGFACE_TOKEN")
if device == "cuda":
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False)
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map=device, token=hg_token).eval()
else:
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, token=hg_token).eval()
processor = AutoProcessor.from_pretrained(model_id, token=hg_token, device_map=device)
prompt = "What is shown in this picture?"


#
# Launch
#

loop(device=device)
window.mainloop()
