import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINE LIGHTWEIGHT LPR NETWORK CLASSIFIER
# ==========================================
class LPRModule(nn.Module):
    def __init__(self):
        super(LPRModule, self).__init__()
        # Simulating a simple character recognition network
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 28 * 28, 10, bias=False) 

        # Initialize weights uniformly so the output is stable and clear
        nn.init.constant_(self.conv.weight, 0.2)
        nn.init.constant_(self.fc.weight, 0.05)

    def forward(self, x):
        self.x0 = x.clone().detach().requires_grad_(True)
        self.x1 = self.conv(self.x0)
        self.x2 = self.relu(self.x1)
        self.x3 = self.flatten(self.x2)
        out = self.fc(self.x3)
        return out

# ==========================================
# 2. CUSTOM ENGINE FOR LAYER-WISE RELEVANCE PROPAGATION
# ==========================================
def lrp_linear(x, w, R_j, eps=1e-9):
    """Propagates relevance backwards through a Linear layer."""
    x_flat = x.view(1, -1)  # Shape: [1, 3136]
    sign = torch.sign
    
    # Transpose weights to align with matrix multiplication: [1, 3136] @ [3136, 10] = [1, 10]
    z = x_flat @ w.weight.t() 
    s = R_j / (z + eps * sign(z))  # Shape: [1, 10]
    
    # Propagate back to input neurons: [1, 10] @ [10, 3136] = [1, 3136]
    c = s @ w.weight
    R_i = x_flat * c
    return R_i.view_as(x)

def lrp_conv2d(x, conv, R_j, eps=1e-9):
    """Propagates relevance backwards through a Conv2d layer."""
    x_activated = x.clone().detach().requires_grad_(True)
    z = conv(x_activated)
    s = (R_j / (z + eps * torch.sign(z))).detach()
    (z * s).backward(torch.ones_like(z))
    R_i = (x_activated.grad * x_activated).detach()
    return R_i

# ==========================================
# 3. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Create a synthetic 28x28 grayscale image mimicking a license plate character (a vertical "I" or "1" stroke)
    mock_plate_char = torch.zeros(1, 1, 28, 28)
    mock_plate_char[0, 0, 5:23, 12:16] = 1.0  
    
    # Run the model forward pass
    model = LPRModule()
    model.eval()
    output = model(mock_plate_char)
    
    target_class = torch.argmax(output, dim=1).item()
    print(f"LPR Model predicted character class index: {target_class}")
    
    # Initialize Relevance vector (R) at the top output layer
    R_top = torch.zeros_like(output)
    R_top[0, target_class] = output[0, target_class] 
    
    # --- Backpropagation of Relevance (LRP Phase) ---
    R_flattened = lrp_linear(model.x2, model.fc, R_top)
    R_conv = R_flattened.view_as(model.x2)
    R_input = lrp_conv2d(mock_plate_char, model.conv, R_conv)
    
    # ==========================================
    # 4. PLOT AND SAVE THE ATTRIBUTION HEATMAP
    # ==========================================
    heatmap = R_input[0, 0].numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(mock_plate_char[0,0].numpy(), cmap='gray')
    axes[0].set_title("Input Plate Crop")
    axes[0].axis('off')
    
    im = axes[1].imshow(heatmap, cmap='Reds')
    axes[1].set_title("LRP Pixels Attribution")
    axes[1].axis('off')
    
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    # Save the output visualization image to disk
    plt.savefig("lrp_output.png")
    print("Success! Heatmap explanation saved to 'lrp_output.png'.")