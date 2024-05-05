

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to("cuda")


raw_image = Image.open('ASCII-Images/ascii_art.png')

question = "Lion"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

itm_scores = model(**inputs)[0]

probabilities = torch.softmax(itm_scores, dim=1)
cosine_score = model(**inputs, use_itm_head=False)[0]

print(itm_scores)
print(probabilities)
print(cosine_score)