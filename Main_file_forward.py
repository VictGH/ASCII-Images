import os
import torch
import json
import re
import ollama
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
os.chdir(os.path.join(os.getcwd(), 'ASCII-Images'))
from CNN_network import TextCNN, ResidualBlock
from ASCII_to_png import ascii_art_to_image
from ALIGN import get_image_probabilities




def split_text_into_batches(text, seq_length = 100):
    """
    Split a given text into batches of specified length, padding the last batch if necessary.
    
    Parameters:
    text (str): The text to split into batches.
    seq_length (int): The length of each batch.
    
    Returns:
    List[str]: A list of text batches.
    """
    n = len(text)
    num_batches = (n + seq_length - 1) // seq_length  # Calculate the total number of batches needed
    batches = [text[i * seq_length: (i + 1) * seq_length] for i in range(num_batches)]
    
    # Pad the last batch if it is less than seq_length
    if len(batches[-1]) < seq_length:
        batches[-1] = (batches[-1]+ [0] * 100)[:100]
    
    return batches

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Use re.findall to capture sequences of lowercase letters or single non-lowercase characters
    tokens = re.findall(r'[a-z]+|[^a-z]', text)
    return tokens

def map_words_to_values(word_dict, words):
    # Check if the values are a list of integer numbers starting from zero and increasing by 1
    values = set(word_dict.values())
    max_value = -1
    for value in values:
        if value > max_value + 1:
            raise ValueError(f"Missing number: {max_value + 1}")
        max_value = max(max_value, value)

    next_value = max_value + 1

    output_list = []
    for word in words:
        if word not in word_dict:
            word_dict[word] = next_value
            next_value += 1
        output_list.append(word_dict[word])

    return output_list

def text_to_batch(text,address):
    """
    Turns a text into a batch of texts of length 100 with each word turned into its corresponding number 
    in the vocabulary dictionary. We also read the dictionary and update it
    
    Parameters:
    text (str): The text to split into batches.
    
    Returns:
    List[str]: A list of text batches.
    """


    try:
        with open(address, 'r') as file:
            dictionary = json.load(file)
       #print(f"Dictionary successfully read from {address}")
    except (IOError, json.JSONDecodeError) as e:
        print(f"An error occurred while reading the dictionary from {address}: {e}")
      

    # Apply modifications to the dictionary using the provided function
    #print('dictionary',dictionary)
    batches = split_text_into_batches(map_words_to_values(dictionary,preprocess_text(text)))
    #print('dictionary',dictionary)
    # Write the modified dictionary back to the file
    try:
        with open(address, 'w') as file:
            json.dump(dictionary, file, indent=4)
        #print(f"Dictionary successfully written to {address}")
    except IOError as e:
        print(f"An error occurred while writing the dictionary to {address}: {e}")

    
    return batches

def apply_dict_efficient(tensor, mapping_dict = {-3: ' ', -2: '.', -1: '-', 0: '+', 1: '#', 2: '@'}):
    """
    Apply a mapping dictionary to each element in a 2D tensor using vectorized operations.
    
    :param tensor: A 2D PyTorch tensor.
    :param mapping_dict: A dictionary where keys are the original values and values are the mapped values.
    :return: A new list of lists with the mapping applied.
    """
    # Detach the tensor and convert to a numpy array
    tensor_np = tensor.detach().numpy()
    
    # Create a new numpy array to hold the mapped values
    result_np = np.empty(tensor_np.shape, dtype=object)
    
    # Vectorized mapping
    for key, value in mapping_dict.items():
        result_np[tensor_np == key] = value
    
    # Convert the result to a list of lists and return
    return result_np.tolist()


#prompt_0 = "You are connected to a deep convolutional neural network. I will take your output and transfrom it into an ascii image. \
#It is important that you just give the necessary input. Do not explain anything to me. Just the necessary \
#output for the convolutional network. Write some text so that the neural network can draw the ascii image of the following description \n"
#description = "A majestic lion sits prominently on a pure white background, its golden fur glistening in the light. The lion's powerful physique and regal expression command attention as it surveys its domain."


def add_element_to_json(file_path, context, prompt, answer, score,topic,item,prompt_index):
    # Read the existing data from the JSON file
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # If the file does not exist, initialize an empty list
        data = []
    
    # Create the new element
    new_element = {
        'context': context,
        'prompt': prompt,
        'answer': answer,
        'score': score,
        'Topic Index': topic,
        'Item Index': item,
        'Prompt Index': prompt_index

    }
    
    # Add the new element to the existing data
    data.append(new_element)
    
    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def prompt_to_proba(dict_path,results_json_path,descriptions_path,size = 1000):

    prompt_0 = "You are connected to a deep convolutional neural network. I will take your output and transfrom it into an ascii image. \
    It is important that you just give the necessary input. Do not explain anything to me. Just the necessary \
    output for the convolutional network. Write some text so that the neural network can draw the ascii image of the following description \n"

    vocab_size = 10000
    embed_size = 300
    num_blocks = 10
    output_shape = (50, 30)
    model = TextCNN(vocab_size, embed_size, num_blocks, output_shape)

    df_description = pd.read_csv(descriptions_path,sep = ';',index_col = False).sample(size)
    for i,row in df_description.iterrows():
        topic = row['Topic Index']
        item = row['Item Index']
        prompt_index = row['Index Prompt']
        description = row['Prompt']

        response = ollama.generate(model='llama3', prompt=prompt_0+description)['response']
        x = text_to_batch(response,dict_path)
        x = torch.tensor(x, dtype=torch.long)
        output = model(x)
        output_ascii = apply_dict_efficient(output)
        png_image = ascii_art_to_image(output_ascii)
        proba = get_image_probabilities(png_image,description)[0][1].item()
        
        print(proba)

        add_element_to_json(results_json_path, prompt_0, description, response, proba,topic,item,prompt_index)
    
    
prompt_to_proba('dict_vocab.json','results_t0.json','descriptions.csv',size = 5000)
