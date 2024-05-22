from PIL import Image, ImageDraw

def ascii_art_to_image(ascii_art):
    """
    Converts ASCII art into an image and displays it.
    
    :param ascii_art: A list of lists containing ASCII art characters.
    """
    # Character to color mapping
    char_to_color = {
        ' ': 'white',   # White
        '.': '#e0e0e0', # Lighter grey
        '-': '#c0c0c0', # Light grey
        '+': '#a0a0a0', # Medium grey
        '#': '#606060', # Dark grey
        '@': 'black'    # Black
    }

    char_width = 8
    char_height = 12

    # Determine the image size
    img_width = char_width * max(len(line) for line in ascii_art)
    img_height = char_height * len(ascii_art)

    # Create a new blank image with a white background
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)

    # Draw each block
    for y, line in enumerate(ascii_art):
        for x, char in enumerate(line):
            # Draw the block with its top-left corner at (x * char_width, y * char_height)
            color = char_to_color[char]
            draw.rectangle([x * char_width, y * char_height, (x+1) * char_width, (y+1) * char_height], fill=color)

    # Resize image to 200x200 pixels
    image = image.resize((200, 200), Image.LANCZOS)

    # Display the image
    image.show()

# Example usage:
ascii_art = [
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', '.', '-', '+', '#', '@', '#', ' '],
    [' ', '.', '-', '+', '#', '#', '#', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
]

ascii_art_to_image(ascii_art)
