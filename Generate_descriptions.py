import ollama
from io import StringIO
import pandas as pd
import numpy as np


def custom_function(example):
    outputs = []
    prompt_0 = f"Give me a definition for a painting of {example}. Yhis definition will be used for prompting a painting program. Give an at most two senteces description.\
    Describe the background as white and just give the descrption. Nothing else. Just write the description. Do not add any note. Do not say if I need any adjustment. Do not introduce\
    with 'here we have a definition' just write the definition and that's all"
    response = ollama.generate(model='llama3',options= { "seed": int(np.random.rand()*1000000),  "temperature": 0.5}, prompt=prompt_0 )
    outputs.append(response['response'])
    combined_output = "\n".join(outputs)
    print('#################')
    print(combined_output)
    return combined_output
    
df = pd.read_csv('real_things_examples.csv',sep = ',', index_col = False)
# Repeat each row in the DataFrame 10 times and create a new index
df_repeated = df.loc[df.index.repeat(10)].reset_index(drop=True)

# Add a new column with numbers from 0 to 9 for each repeated group
df_repeated['Index Prompt'] = df_repeated.groupby(df_repeated.index // 10).cumcount()

# Apply the custom function to each row and create a new column
# Ensuring the function uses the 'Index' and other relevant data from each row
df_repeated['Prompt'] = df_repeated.apply(lambda row: custom_function(row['Example']), axis=1)


df_repeated.to_csv('descriptions.csv',sep = ';',index = False)