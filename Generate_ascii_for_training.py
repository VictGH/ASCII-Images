import ollama
import pandas as pd
import numpy as np
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import time
import gzip

df = pd.read_csv('descriptions.csv',sep  = ';',index_col = False)
print('a')
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
pipe.to("cuda")

batch_data = []
filename = 'ascii_art.gz'

def image_to_ascii_optimized(image, width=100, height=45, gradient=" .-+#@"):
    # Resize the image to the target dimensions
    image = image.resize((width, height))
    
    # Convert the image to grayscale
    gray_image = image.convert("L")
    
    # Convert the grayscale image to a numpy array
    image_array = np.array(gray_image)
    
    # Normalize the pixel values to use integer operations
    factor = len(gradient) / 256
    # Using numpy to directly convert the pixel values to indices in the gradient
    ascii_indices = (image_array * factor).astype(int)
    
    # Create the ASCII image
    ascii_image = "\n".join(
        "".join(gradient[idx] for idx in row) for row in ascii_indices
    )
    
    return ascii_image

def generate_n_images(df,row,n = 100, batch_size = 1000):

    for i,row in df[row:row+1].iterrows():

        prompt = row['Prompt']

        # Generate and save 100 images
        for i in range(n):
            result = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0)
            ascii_art = image_to_ascii_optimized(result.images[0])

            image_id =f"{row['Topic Index']}_{row['Item Index']}_{row['Example']}_{row['Index Prompt']}_{i}"

            add_ascii_image_to_batch(image_id,ascii_art,batch_size)



def add_ascii_image_to_batch(image_id, ascii_art, batch_size= 1000):
    global batch_data
    # Append the current ASCII art with its ID to the batch
    batch_data.append(f'ID: {image_id}\n{ascii_art}\n<END_OF_IMAGE>\n')
    
    # Check if the batch has reached the specified size
    if len(batch_data) >= batch_size:
        write_batch_to_file()

def write_batch_to_file():
    global batch_data
    try:
        # Open the gzip file in append mode and write the batch data
        with gzip.open(filename, 'at', compresslevel=5) as file:
            file.writelines(batch_data)
        # Clear the batch data after writing
        batch_data = []
    except IOError as e:
        print(f"An error occurred while writing to file: {e}")


def flush_remaining_data():
    # Call this function at the end of all operations to ensure all remaining data is written
    if batch_data:
        write_batch_to_file()

#for i in range(50,1000): 
#    generate_n_images(df,i,100, 500)

import gzip

def read_and_print_ascii_art(filename='ascii_art.gz'):
    try:
        with gzip.open(filename, 'rt') as file:  # Open in read and text mode
            content = file.read()
            ascii_images = content.split('<END_OF_IMAGE>\n')
            for ascii_image in ascii_images[-5:]:
                print('a')
                if ascii_image:  # Check if there's any content to avoid printing empty lines
                    print(ascii_image)
                    print("---------- Next Image ----------\n")
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")

#Call the function to read and display the ASCII art
read_and_print_ascii_art()
