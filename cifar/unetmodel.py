import torch.nn as nn
from abc import abstractmethod
import math
import torch
from blocks import AttentionBlock, Downsample, Residual_block, TimestepEmbedSequential, Upsample
from utils import norm_layer, timestep_embedding


class UnetModel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 model_channels=128,
                 out_channels=3,
                 num_res_blocks=2,
                 attention_resolutions=(8,16),
                 dropout=0,
                 channel_mult=(1,2,2,2),
                 conv_resample=True,
                 num_heads=4,
                 class_num=10
                ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.class_num = class_num
        
        #time embedding
        time_emb_dim = model_channels*4
        self.time_emb = nn.Sequential(
                nn.Linear(model_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        #class embedding
        class_emb_dim = model_channels
        self.class_emb = nn.Embedding(class_num, class_emb_dim)
        
        #down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_channels = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [Residual_block(ch, model_channels*mult, time_emb_dim, class_emb_dim, dropout)]
                ch = model_channels*mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_channels.append(ch)
            if level != len(channel_mult)-1: # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_channels.append(ch)
                ds*=2
                
        #middle blocks
        self.middle_blocks = TimestepEmbedSequential(
            Residual_block(ch, ch, time_emb_dim, class_emb_dim, dropout), 
            AttentionBlock(ch, num_heads),
            Residual_block(ch, ch, time_emb_dim, class_emb_dim, dropout)
        )
        
        #up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in enumerate(channel_mult[::-1]):
            for i in range(num_res_blocks+1):
                layers = [
                    Residual_block(ch+down_block_channels.pop(), model_channels*mult,\
                                   time_emb_dim, class_emb_dim, dropout)]
                ch = model_channels*mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if level!=len(channel_mult)-1 and i==num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))
                
        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, timesteps, c, mask):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param c: a 1-D batch of classes.
        :param mask: a 1-D batch of conditioned/unconditioned.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step and class embedding
        t_emb = self.time_emb(timestep_embedding(timesteps, dim=self.model_channels))
        c_emb = self.class_emb(c)
        
        
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, t_emb, c_emb, mask)
#             print(h.shape)
            hs.append(h)
        
        # middle stage
        h = self.middle_blocks(h, t_emb, c_emb, mask)
        
        # up stage
        for module in self.up_blocks:
#             print(h.shape, hs[-1].shape)
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, t_emb, c_emb, mask)
        
        return self.out(h)
    