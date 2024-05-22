from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch

def get_image_probabilities(raw_image, question):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to("cuda")


    inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

    itm_scores = model(**inputs)[0]
    probabilities = torch.softmax(itm_scores, dim=1)
    #cosine_score = model(**inputs, use_itm_head=False)[0]

    return probabilities

