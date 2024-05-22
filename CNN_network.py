import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_blocks, output_shape):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.entry_conv = nn.Conv2d(1, 16, kernel_size=(7, 7), stride=(1, int(embed_size/30)), padding=(3, 0))
        self.res_blocks = nn.Sequential(*[ResidualBlock(16) for _ in range(num_blocks)])
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_shape)  # Reshape to target output shape
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.batch_pool = nn.AdaptiveAvgPool2d((1, 30))  # New pooling layer to reduce batch dimension

    def forward(self, x):
        x = self.embedding(x)
        #print(x.shape)
        x = x.unsqueeze(1)  # Channel dimension
        #print(x.shape)
        x = self.entry_conv(x)
        #print(x.shape)
        x = self.res_blocks(x)
        #print(x.shape)
        x = self.adaptive_pool(x)  # Apply adaptive pooling to fix spatial dimensions
        #print(x.shape)
        x = x.permute(2, 1, 0, 3)  # Rearrange dimensions to: [batch, width, channels, height]
        #print(x.shape)
        x = self.batch_pool(x)  # Pool across the batch dimension to aggregate all examples
        #print(x.shape)
        x = x.permute(2, 1, 3, 0)  # Return dimensions to: [1, channels, height, width]
        #print(x.shape)
        x = self.final_conv(x)
        #print(x.shape)
        x = torch.squeeze(x, dim=0)  # Remove the first dimension if it is 1
        x = torch.squeeze(x, dim=0)
        return torch.clamp(torch.round(x), -3, 2)

