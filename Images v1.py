from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import time

# Load the model
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
pipe.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

# Start timing
start_time = time.time()

# Generate and save 100 images
for i in range(100):
    result = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0)
    image = result.images[0]
    # Save the PIL Image directly
    image.save(f"image_{i+1}.png")

# End timing
end_time = time.time()

# Calculate the total execution time
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")

# Optional: Print how many images have been saved
print(f"100 images have been saved.")
