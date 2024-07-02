from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import sys, os

# prep
arguments = sys.argv[1:]
if len(arguments) != 2:
    script_name = os.path.basename(__file__)
    print(f"Usage: python {script_name} <image-path> <quoted-prompt>")
    exit()
image_path = arguments[0]
prompt = arguments[1]

print("Image path: ", image_path)
print("Prompt: ", prompt)

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

print("Thinking...")

image = Image.open(image_path)
inputs = processor(prompt, image, return_tensors="pt")
for i in range(20):
    output = model.generate(**inputs, max_new_tokens=20)
    print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])

