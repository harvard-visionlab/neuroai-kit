import torch
import torch.nn as nn
from torchvision import models
from pdb import set_trace

__all__ = ['NeuroActivationEditor', 'generate_mask']

class NeuroActivationEditor(nn.Module):
    '''
        NeuroActivationEditor class allows you to edit outputs of specific layers in the model.
        This class uses PyTorch's forward hooks to modify the outputs of layers by applying a mask.

        It is designed to handle outputs that are single tensors, as well as outputs that are tuples or lists of tensors,
        making it flexible for a wide range of layers.

        It can be used as a context manager to automatically clean up hooks after use.
    '''
    def __init__(self, model, layers, masks, return_activations=False, detach=True, clone=True, device='cpu'):
        super().__init__()
        self.model = model
        self.layers = [layers] if isinstance(layers, str) else layers
        self.masks = [masks] if not isinstance(masks, list) else masks
        self.return_activations = return_activations
        self.detach = detach
        self.clone = clone
        self.device = device
        self._activations = {layer: torch.empty(0) for layer in self.layers}
        self.hooks = {}

    def hook_layers(self):        
        self.remove_hooks()
        for layer_id, mask in zip(self.layers, self.masks):
            layer = dict([*self.model.named_modules()])[layer_id]
            self.hooks[layer_id] = layer.register_forward_hook(self.edit_outputs_hook(layer_id, mask))
    
    def remove_hooks(self):
        for layer_id in self.layers:
            self._activations[layer_id] = torch.empty(0)
            if layer_id in self.hooks:
                self.hooks[layer_id].remove()
                del self.hooks[layer_id]
    
    def __enter__(self, *args): 
        self.hook_layers()
        return self
    
    def __exit__(self, *args): 
        self.remove_hooks()

    def edit_outputs_hook(self, layer_id, mask):
        # Helper functions to handle various output types
        def detach(output):
            if isinstance(output, tuple): return tuple([o.detach() for o in output])
            elif isinstance(output, list): return [o.detach() for o in output]
            else: return output.detach()

        def clone(output):
            if isinstance(output, tuple): return tuple([o.clone() for o in output])
            elif isinstance(output, list): return [o.clone() for o in output]
            else: return output.clone()

        def to_device(output, device):
            if isinstance(output, tuple): return tuple([o.to(device) for o in output])
            elif isinstance(output, list): return [o.to(device) for o in output]
            else: return output.to(device)
        
        # The hook function applied to the output of the layer
        def fn(_, __, output):
            if isinstance(output, (tuple, list)):
                # If the output is a tuple or list, apply mask to each element
                modified_output = [o * mask for o in output]
            else:
                # Apply mask directly if the output is a single tensor
                modified_output = output * mask

            # Optionally detach, clone, and move to device
            if self.detach: modified_output = detach(modified_output)
            if self.clone: modified_output = clone(modified_output)
            if self.device: modified_output = to_device(modified_output, self.device)

            self._activations[layer_id] = modified_output
            return modified_output
        
        return fn

    def forward(self, x, return_activations=None):
        return_activations = self.return_activations if return_activations is None else return_activations
        out = self.model(x)
        if return_activations:
            return self._activations
        else:
            return out

def generate_mask(shape, units):
    mask = torch.ones(shape).flatten()
    mask[units] = 0
    return mask.reshape(shape).unsqueeze(0)  