import torch
import torch.nn as nn
import warnings
from pdb import set_trace

__all__ = ['NeuroElectrodeArray', 'get_layers', 'get_layer_names', 'get_layer_type', 'get_layer_shapes', 'get_activations']

class NeuroElectrodeArray(nn.Module):
    '''
        NeuroElectrodeArray class that allows you to retain outputs of any layer.
        This class uses PyTorch's "forward hooks", which let you insert a function
        that takes the input and output of a module as arguements.        
        
        In this hook function you can insert tasks like storing the intermediate values,
        or as we'll do in the NeuroEditor class, actually modify the outputs.
        Adding these hooks can cause headaches if you don't "remove" them 
        after you are done with them. For this reason, the FeatureExtractor is 
        setup to be used as a context, which sets up the hooks when
        you enter the context, and removes them when you leave:
        
        with NeuroElectrodeArray(model, layer_name) as electrode:
            activations = electrode(imgs)
            
        If there's an error in that context (or you cancel the operation),
        the __exit__ function of electrode array is executed,
        which we've setup to remove the hooks. This will save you 
        headaches during debugging/development.

        For a more comprehensive, full-feature extractor (e.g., one that
        can pull out internal computations), try torchlens (https://github.com/johnmarktaylor91/torchlens).
    '''    
    def __init__(self, model, layers, retain=True, detach=True, clone=True, device='cpu'):
        super().__init__()
        layers = [layers] if isinstance(layers, str) else layers
        self.model = model
        self.layers = layers
        self.detach = detach
        self.clone = clone
        self.device = device
        self.retain = retain
        self._activations = {layer: torch.empty(0) for layer in layers}        
        self.hooks = {}
        
    def hook_layers(self):        
        self.remove_hooks()
        for layer_id in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            self.hooks[layer_id] = layer.register_forward_hook(self.save_outputs_hook(layer_id))
    
    def remove_hooks(self):
        for layer_id in self.layers:
            if self.retain == False:
                self._activations[layer_id] = torch.empty(0)
            if layer_id in self.hooks:
                self.hooks[layer_id].remove()
                del self.hooks[layer_id]
    
    def __enter__(self, *args): 
        self.hook_layers()
        return self
    
    def __exit__(self, *args): 
        self.remove_hooks()
            
    def save_outputs_hook(self, layer_id):
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
        def fn(_, __, output):
            if self.detach: output = detach(output)
            if self.clone: output = clone(output)
            if self.device: output = to_device(output, self.device)
            self._activations[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._activations
    
def get_layers(model, parent_name=''):
    layer_info = []
    for module_name, module in model.named_children():
        layer_name = parent_name + '.' + module_name
        if len(list(module.named_children())):
            # Recursively accumulate layers
            layer_info.extend(get_layers(module, layer_name))
        else:
            layer_info.append(layer_name.strip('.'))
    
    return layer_info

def get_layer_names(model):
    return get_layers(model, parent_name='', layer_info=[])

def get_layer_type(model, layer_name):
    for name,m in list(model.named_modules()):
        if name == layer_name: return m.__class__.__name__

@torch.no_grad()
def get_layer_shapes(model, layer_names, x):
    model.eval()
    with NeuroElectrodeArray(model, layer_names) as electrode:
        activations = electrode(x)
        shapes = {k:v.shape for k,v in activations.items()}
    return shapes

@torch.no_grad()
def get_activations(model, imgs, layer_names=None):
    if model.training:
        warnings.warn("Warning, you are running your model in 'train' mode. You should probably use model.eval()")

    with NeuroElectrodeArray(model, layer_names) as electrode:
        activations = electrode(imgs)

    return activations