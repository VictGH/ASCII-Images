import ollama
from io import StringIO
import pandas as pd

real_things = [
    "Animals", "Plants", "Fruits", "Vegetables", "Birds", "Fish", "Trees", "Flowers",
    "Mammals", "Insects", "Reptiles", "Amphibians", "Fungi", "Minerals", "Rocks",
    "Mountains", "Rivers", "Oceans", "Lakes", "Deserts", "Forests", "Cities", "Countries",
    "Musical Instruments", "Songs", "Books", "Paintings", "Sculptures", "Photographs",
    "Films", "Costumes", "Fabrics", "Clothes", "Shoes", "Hats", "Jewelry", "Watches",
    "Cars", "Motorcycles", "Bicycles", "Airplanes", "Trains", "Ships", "Spacecraft",
    "Electronics", "Computers", "Video Games", "Board Games", "Sports Equipment",
    "Festivals", "Religious Artifacts", "Scientific Equipment", "Chemical Elements",
    "Chemical Compounds", "Medications", "Medical Instruments", "Tools", "Home Appliances",
    "Furniture", "Buildings", "Architectural Models", "Landmarks", "Historical Artifacts",
    "Food Items", "Beverages", "Spices", "Cooking Implements", "Recipes", "Restaurants",
    "Schools", "Universities", "Space Suits", "Crystals", "Dolls", "Toys", "Statues",
    "Bridges", "Highways", "Islands", "Coral Reefs", "Caves", "Helmets", "Gloves", "Glasses",
    "Mirrors", "Telescopes", "Microscopes", "Sewing Machines", "Pianos", "Drums", "Candles",
    "Blankets", "Pots and Pans", "Cutlery", "Baskets", "Vases", "Notebooks", "Pens",
    "Paintbrushes", "Sculpting Tools"
]
outputs = []
for i,thing in enumerate(real_things):
    print(i,thing)
    prompt_0 = f"Give me ten different examples of the topic {thing}. They all must be different and fairly common. Also, write it in this csv like format,\
    {i};j;example where {i} is always this number I am writing and j goes from 0 to 9 and example is the example you chose. Write only the csv part, no more answers, just like a csv.\
    I repeat, the first number is always the same. The second one goes from 0 to 9"
    response = ollama.generate(model='llama3', prompt=prompt_0)
   
    outputs.append(response['response'])

# Combine all outputs into one string
combined_output = "\n".join(outputs)

# Use StringIO to simulate reading from a file
data = StringIO(combined_output)

# Create DataFrame
df = pd.read_csv(data, header=None, names=['Topic Index', 'Item Index', 'Example'], sep=';')

# Save DataFrame to CSV file
df.to_csv('real_things_examples.csv', index=False)

# Print DataFrame to verify
print(df)
