import torch

class OneLayer(torch.nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size):
        super().__init__()
        padding = (kernel_size-1) // 2
        self.weights = torch.nn.Conv2d(input_nc, output_nc, kernel_size, padding=padding)
        print('Simple one-layer Conv2d with input_nc (%d), output_nc (%d), kernel_size (%d, %d) was created.' % (input_nc, output_nc, kernel_size, kernel_size))

    def forward(self, x):
        return self.weights(x)
        
