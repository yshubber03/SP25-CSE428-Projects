#adopted from https://github.com/facebookresearch/mae repo


from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

from model.pos_embed import convert_count_to_pos_embed_cuda
from model.SmallSequenceCNN import SmallSequenceCNN

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,  **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.patch_size = kwargs['patch_size']
        self.in_chans = kwargs['in_chans']
        self.embed_dim = kwargs['embed_dim']
        self.use_sequence = True

        # Override positional embedding to add +1 for sequence token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1 + 1 + 1, self.embed_dim)
        )


        # Add the sequence encoder
        self.sequence_encoder = SmallSequenceCNN(input_channels=4, embed_dim=self.embed_dim)

    def forward_features(self, x,total_count, sequence_input):
        B = x.shape[0]

        # Patch embedding (image input)
        x = self.patch_embed(x)

        ## total count stuff ###
        total_count = torch.log10(total_count)
        count_embed = convert_count_to_pos_embed_cuda(total_count, self.embed_dim)
        count_embed = count_embed.unsqueeze(1)# (N, 1, D)

        # Sequence embedding (new addition)
        sequence_embed = self.sequence_encoder(sequence_input).unsqueeze(1)  # (B, 1, D)
        
        # === DEBUG LOGGING HOOK ===
        # if not self.training:
        #     # Only log during evaluation or always???? not really sure how the hooks work
        #     print(f"[DEBUG] sequence_embed mean: {sequence_embed.mean().item():.6f}, std: {sequence_embed.std().item():.6f}")

        # if sequence_embed.requires_grad:
        #     sequence_embed.register_hook(grad_hook_fn)

        # Class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # Combine → [CLS] + [SEQ] + [PATCHES]
        x = torch.cat((cls_tokens, count_embed, sequence_embed, x), dim=1)  # (B, 1+1+num_patches, D)

        # Check that positional embeddings match
        if self.pos_embed.shape[1] != x.shape[1]:
            raise ValueError(f"pos_embed shape mismatch: expected {x.shape[1]}, got {self.pos_embed.shape[1]}")

        # Add positional embeddings
        x = x + self.pos_embed

        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        
        x = self.norm(x)

        return x
    
def grad_hook_fn(grad):
    print(f"[GRADIENT HOOK] sequence_embed grad mean: {grad.mean().item():.6f}, std: {grad.std().item():.6f}")

def vit_base_patch16(**kwargs):
    model = VisionTransformer(in_chans=3,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(in_chans=3,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(in_chans=3,
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
