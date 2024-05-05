import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()  # Correct and simplified usage of super() in Python 3
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        print("ResBlock Input:", x.shape)
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        print("ResBlock out:", out.shape)
        return F.relu(out)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_blocks, output_shape):
        super().__init__()  # Using super() correctly in Python 3
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.entry_conv = nn.Conv2d(1, 16, kernel_size=(7, 7), stride=(1, int(embed_size/30)), padding=(3, 0))
        self.res_blocks = nn.Sequential(*[ResidualBlock(16) for _ in range(num_blocks)])
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.output_shape = output_shape

    def forward(self, x):
        x = self.embedding(x)  # Shape: [batch_size, sequence_length, embed_size]
        print("After embedding:", x.shape)
        x = x.unsqueeze(1)     # Add channel dimension: [batch_size, 1, sequence_length, embed_size]
        print("After unsqueeze:", x.shape)
        x = self.entry_conv(x) # Shape after conv: [batch_size, 16, sequence_length, 1]
        print("After entry_conv:", x.shape)
        #x = x.squeeze(3)       # Correctly remove the last singleton dimension: [batch_size, 16, sequence_length]
        #print("After squeeze:", x.shape)
        #x = x.unsqueeze(0)     # Add channel dimension: [1,batch_size, 16, sequence_length]
        #print("After unsqueeze:", x.shape)
        x = self.res_blocks(x) # Process through residual blocks
        print("After res_blocks:", x.shape)
        x = self.final_conv(x) # Reduce channels to 1: [batch_size, 1, sequence_length]
        print("After final_conv:", x.shape)
        x = F.interpolate(x, size=self.output_shape, mode='nearest')  # Resize to the output shape
        print("After interpolate:", x.shape)
        return torch.clamp(torch.round(x), -3, 2)  # Discretize and clamp the output


# Parameters and model instantiation
vocab_size = 10000  # Assume a vocabulary size of 10000
embed_size = 300    # Common embedding size
num_blocks = 10     # Number of residual blocks
output_shape = (50, 30)  # Output dimensions
model = TextCNN(vocab_size, embed_size, num_blocks, output_shape)

# Example input (batch of 10 sequences of length 100)
x = torch.randint(0, vocab_size, (1, 120))
print(x)
output = model(x)
print(output)  # Should print torch.Size([10, 1, 50, 30])
