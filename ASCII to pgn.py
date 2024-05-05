from PIL import Image, ImageDraw, ImageFont

# ASCII Art input as a string, ensure your art is correctly formatted as a string.
ascii_art = """
                             .  ..-+######++++++++------...
                           ..-..++#########+++++--.-------........
                     ...  ..---+###@#####++-----.-----------++++-..
                  ....--...--+++###@####++-++--------------++++--...
                .----..------++++#######++-+---------+++++++++++++++--..
              .------------++++++#######++--+-----++++##########+++++--..
             .-+----------+++++#++##+++++++------++#####++++++#@##++++++--...
             .-++-----+++++++++++######++++-+++++++++##++##+#+++@######++--..
             .-+-----+++++#++---++++++---++++--++++#########++#+#@@####+++++---..
              .------+++++-..--++++---...--+---++#######@@#####+#@@###+++++++++--.
               .------+##-...--+++--....----+#+++#####@@@@#####@@@@###++++++##++--..
                 ---++++.....-++-......------+++-----+##@@#+++@@@@#########++++++++-..
                .-++++-...--+++-...---.--------++--+---+###+++#@@##########++++++++++--..
                .-++----------------+++--.------+---------+++#@@#######+++++++++++++###+-..
                 .+-+##+--.....---.+@####+++--------+-+--....-#########++++++#+++++####++-.
                .---.++-......--. -##+--++++-------+++-------..-+##+++++++++++++++++++##++++-.
              .----++.--....-----++-.-------+++++++++++-++-------+#+++++++++++++++++++++++++-..
              --++------------++++-----+++++++++++++++++++++-----+++++++++++++++++++++++++++#+-.
             .---+--------++--+++++-++++++++++-++++++++++-------.-+++++-++++++++++++++++++++####-.
            .---....--------+++++++----+-----------+-++++----------+++++++--++--++++##+++++++++++-
            .---++---+++-++--+++++++------------------++++++-------++++---+--++--+++++++#++++++###+.
            .---++###@@@##+--+-++--++--....----------++---------.--+++++--+++++++++++++++++++++++###
           ..---..-+##+-.--------------.---.-------+++++++---------++-+++++++--+++++++++++++++++++##
           ..---.. .+. ....---------...------------++++-----------+++--++++--++--+++++++++##++++####
           ..---.   -+-......-------...-.--+------++--------------+++--+++++---++++++--+++++++++##++
           ....--..-###++--.----++-------++++---------------------+++----+++++--++++++++####++++#+++
           ....----...-------++######++++++-----------------.----+#+#+---+++++++++#+++++####++++++++
           ...----    ..... ...+########++--------..------------+++++#+++++++++++++######+++++++++++
           ...---... ..........--+#####+-------.----.---------+++++++++++++#+++#++######++++++++++-+
          ........---------++#+#######+++------..----..-----++++++++++++#++###########++++++++++++++
           ......-----++++++#####++##+--+--.----.-----.-----+##++###+#++##########+###++++++++++++++
           .....-...---+++++++##++##++--+--..--+-.------++++###########+##########++++++++++++++++++
          .--..-...-----++++++++++++++---------+-.----+++########################+++++++++++++++++++
          .+++-...----.--+++++++++++++----.---+++--++++##@######################++++++++++++++++++++
         .+###-------..-+++++++++++-+----..--++#+++#####@#####################++++++++++++++++++++++
         -####+#++--+-++---+++-++++---++---++###############################++++++++++++++++++++++++
        .+++########++++--+++++++----++--++###############################+++++++++++++++++++++++++-
       .--++########++++--++++++-------+#################################+++++++++++++++++++++++++++
       ...++########++++++++++++++++-++#################################++++++++++++++++++++++++++++
        -+##############+++++##++++++#################################+++-++++++++++++++++++++++++++
      .-+#++###########++++####+++++#################################++--+++++++++++++++++++++++++++
     ......--+###########++++--.............--++++##############++#+++----++++++++++++++++++++++++++
------..------+#####++--.......................-----++++#####+++--++++--++++++++-----+++++++++++++++
---.-----++---....... ....------------------------------+++++-------+-++-+++++++---------++---++++++
..----+---.............-----------------------------------+----------+++-+#++++++--+-----++-+++--+++
"""

# Split the ASCII art into lines
lines = ascii_art.strip().split('\n')



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
img_width = char_width * max(len(line) for line in lines)
img_height = char_height * len(lines)

# Create a new blank image with a white background
image = Image.new('RGB', (img_width, img_height), 'white')
draw = ImageDraw.Draw(image)

# Draw each block
for y, line in enumerate(lines):
    for x, char in enumerate(line):
        # Draw the block with its top-left corner at (x * char_width, y * char_height)
        color = char_to_color[char]
        draw.rectangle([x * char_width, y * char_height, (x+1) * char_width, (y+1) * char_height], fill=color)

# Resize image to 50x50 pixels
image = image.resize((200, 200), Image.LANCZOS)


# Display or save the image
image.show()  # Or use image.save('output.png') to save the image


# Save the image
image.save('ASCII-Images/ascii_art.png')