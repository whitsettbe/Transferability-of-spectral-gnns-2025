import torch

# Utility function to initialize weights and biases for a layer
# (returning Parameters for use in PyTorch models)
# based on torch.Linear initialization
def param_weights_and_biases(in_shape, out_shape, bias=True, double=False):
    """
    Initialize weights and biases for a layer.
    
    Args:
        in_shape (int): Number of input features.
        out_shape (int): Number of output features.
        bias (bool): Whether to include a bias term.
    
    Returns:
        tuple: Initialized weights and biases.
    """
    weight = torch.nn.Parameter(torch.empty(in_shape + out_shape).to(torch.float64 if double else torch.float32))
    torch.nn.init.kaiming_uniform_(weight, a=5**.5)
    if bias:
        fan_in = torch.prod(torch.tensor(in_shape)).item()
        bound = 1 / fan_in**.5
        bias = torch.nn.Parameter(torch.empty(out_shape).to(torch.float64 if double else torch.float32))
        torch.nn.init.uniform_(bias, -bound, bound)
    else:
        bias = None
    return weight, bias