################################################################################
## Embedder code from:                                                        ##
## https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py ##
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import functools

PI = 3.141592


# Positional encoding (section 5.1 of the NERF paper)
# note that the embedder takes inputs in the range [-1,1]
class EmbedderNERF:
    def __init__(self, input_dims=3, include_input=True, max_freq_log2=10-1, num_freqs=10, log_sampling=True, periodic_fns=[torch.sin, torch.cos]):
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns

        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs
        
        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], 1)


def get_embedder_nerf(multires, input_dims=3, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'input_dims' : input_dims,
                'include_input' : True,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = EmbedderNERF(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(PI * x)
    return embed, embedder_obj.out_dim