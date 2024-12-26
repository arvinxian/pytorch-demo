import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinaryConvolution(nn.Module):
    def __init__(self, kernel):
        super(BinaryConvolution, self).__init__()
        # Reshape kernel for PyTorch Conv2d (out_channels, in_channels, height, width)
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)
        # Initialize convolution layer with the given kernel
        self.conv = nn.Conv2d(1, 1, kernel_size=self.kernel.shape[-2:], bias=False, padding='same')
        self.conv.weight = nn.Parameter(self.kernel)
    
    def forward(self, x):
        # Add batch and channel dimensions
        x = x.unsqueeze(0).unsqueeze(0)
        # Perform convolution
        conv_output = self.conv(x)
        # Binarize output (values > 1 become 1, others become 0)
        binary_output = (conv_output > 1).float()
        # Remove extra dimensions
        return binary_output.squeeze()

def process_matrix(input_matrix, kernel, padding='valid'):
    """
    Process a binary matrix with convolution and binarization
    
    Args:
        input_matrix: Input binary matrix (0-1 matrix)
        kernel: Convolution kernel
        padding: 'valid' or 'same' padding
    
    Returns:
        Processed binary matrix
    """
    # Convert inputs to tensors
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32)
    
    # Create and apply model
    model = BinaryConvolution(kernel_tensor)
    with torch.no_grad():
        output = model(input_tensor)
    
    return output

# Complete example with test cases
def main():
    # Test case 1: Simple 4x4 matrix with 2x2 kernel
    input_matrix1 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    
    kernel1 = [
        [0.5, 0.6],
        [0.7, 0.8]
    ]
    
    print("Test Case 1:")
    print("Input Matrix:")
    print(np.array(input_matrix1))
    print("\nKernel:")
    print(np.array(kernel1))
    
    result1 = process_matrix(input_matrix1, kernel1)
    print("\nOutput Matrix A:")
    print(result1.numpy())
    
    # Test case 2: Larger matrix with different kernel
    input_matrix2 = [
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1]
    ]
    
    kernel2 = [
        [0.3, 0.4, 0.3],
        [0.4, 0.5, 0.4],
        [0.3, 0.4, 0.3]
    ]
    
    print("\nTest Case 2:")
    print("Input Matrix:")
    print(np.array(input_matrix2))
    print("\nKernel:")
    print(np.array(kernel2))
    
    result2 = process_matrix(input_matrix2, kernel2)
    print("\nOutput Matrix A:")
    print(result2.numpy())

if __name__ == "__main__":
    main()
