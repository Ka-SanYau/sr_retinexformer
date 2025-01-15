import torch

# Define function
def generate_pe_and_means(batch_size, height, width, d_model=64):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate a random tensor to simulate `rgb_255`
    # We'll use random integers in the range [0, 255] to simulate RGB values
    rgb_255 = torch.randint(0, 256, (batch_size, 3, height, width), dtype=torch.int32).to(device)
    channel = 0  # Example: Use the first channel (R)

    # Initialize the position encoding tensor
    pe = torch.zeros((batch_size, d_model, height, width), device=device)
    
    # Compute position encodings
    for b in range(batch_size):
        for i in range(d_model):
            div_term = 10000 ** (i / d_model)
            if i % 2 == 0:
                pe[b, i] = (
                    (1 / 2) * torch.sin(rgb_255[b, channel, :, :] / div_term) + 0.5
                )
            else:
                pe[b, i] = (
                    (1 / 2) * torch.cos(rgb_255[b, channel, :, :] / div_term) + 0.5
                )

    # Compute means for the grouped channels
    p1 = pe[:, 0:16].mean(dim=1, keepdim=True)  # Mean across channels 0-15
    p2 = pe[:, 16:32].mean(dim=1, keepdim=True)  # Mean across channels 16-31
    p3 = pe[:, 32:48].mean(dim=1, keepdim=True)  # Mean across channels 32-47
    p4 = pe[:, 48:64].mean(dim=1, keepdim=True)   # Mean across channels 48-63

    return rgb_255, pe, p1, p2, p3, p4

# Parameters
batch_size = 2
height = 4
width = 4
d_model = 64

# Generate and compute
rgb_255, pe, p1, p2, p3, p4 = generate_pe_and_means(batch_size, height, width, d_model)

# Print results
print("Random RGB Tensor (rgb_255):")
print(rgb_255)
print("\nGenerated Position Encoding Tensor (pe):")
print(pe)
print("\nMean of Channels 0-15 (p1):")
print(p1)
print("\nMean of Channels 16-31 (p2):")
print(p2)
print("\nMean of Channels 32-47 (p3):")
print(p3)
print("\nMean of Channels 48-63 (p4):")
print(p4)