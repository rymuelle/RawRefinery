from RawRefinery.model.Cond_NAFNet import ConditionedChannelAttention, SimpleGate
import torch
import torch.nn as nn
import numpy as np

class LKA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        squeeze_chans = channels//4
        self.in_conv = nn.Conv2d(channels, squeeze_chans, 1, 1, 0)
        self.large_conv = nn.Conv2d(squeeze_chans, squeeze_chans, 7, 1, 3, groups=squeeze_chans)
        self.out_conv = nn.Conv2d(squeeze_chans, channels, 1, 1, 0)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.large_conv(x)
        x = self.out_conv(x)
        return x 
class GatedBottleNeck(nn.Module):
    def __init__(self, chan, ratio=2, cond_chans=0, r=16):
        super().__init__()
        interior_chan = int(chan * ratio)
        self.pwconv = nn.Conv2d(chan, interior_chan, 1, 1, 0)
        self.dwconv = nn.Conv2d(interior_chan, interior_chan, 3, 1, 1, groups=interior_chan)
        self.lka =  LKA(interior_chan)
        
        self.act = SimpleGate()
        self.sca = ConditionedChannelAttention(interior_chan // 2, cond_chans)
        self.pwconv2 = nn.Conv2d(interior_chan // 2 , chan, 1, 1, 0)
        

        
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.norm = nn.GroupNorm(1, chan)
    def forward(self, input):
        in_x = input[0]
        in_cond = input[1]
        x = self.norm(in_x)
        
        x = self.pwconv(x)
        x = self.lka(x)*x
        x = self.dwconv(x)
        x = self.act(x)
        x = self.sca(x, in_cond) * x

        x = self.pwconv2(x)   

        return (in_x + self.beta*x, in_cond)
    

class ScaleAndProcess(nn.Module):
    def __init__(self, in_chan, width, out_chan, n_blocks, scale, cond_chans=1):
        super().__init__()
        assert np.log(scale)/np.log(2) == int(np.log(scale)/np.log(2)), f"Scale ({scale}) must be a multiple of 2"
        padding = max(scale // 2 - 1, 0)
        self.scale = scale
        self.in_conv = nn.Conv2d(in_chan, width, scale, scale, padding)
        self.blocks = nn.Sequential(
            *[BottleNeck(width, cond_chans=cond_chans) for _ in range(n_blocks)]
        )
        self.out_conv = nn.Conv2d(width, out_chan, 1, 1, 0)
    
    def forward(self, input):
        if len(input) == 2:
            in_image, cond = input
        elif len(input)==3:
            in_image, cond, previous_output = input
        after_conv = self.in_conv(in_image)
        if len(input) == 3:
            after_conv = after_conv + previous_output
            
        output = self.blocks((after_conv, cond))[0]
        
        reduced_image = torch.nn.functional.interpolate(in_image, scale_factor=1/self.scale)
        out_image = self.out_conv(output) + reduced_image
        
        return output, out_image, reduced_image

from time import perf_counter
class BigDecoder(nn.Module):
    def __init__(self, in_chan, out_chan, widths=[64, 64, 64, 64], blocks=[8,8,8,8], cond_chans=1):
        super().__init__()
        self.stages = nn.ModuleList()
        self.ups = nn.ModuleList()
        scale = 2**(len(blocks)-1)

        for current_stage,(width, block) in enumerate(zip(widths, blocks)):
            self.stages.append(ScaleAndProcess(in_chan, width, out_chan, block, scale, cond_chans=cond_chans))
            scale = scale // 2
            if len(blocks) > current_stage+1:
                next_width = widths[current_stage+1]
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(width, next_width * 4, 1, bias=False), nn.PixelShuffle(2)
                    )
                )

    def forward(self, img, cond):
        out_images = []
        input = (img, cond)
        for current_stage, stage in enumerate(self.stages):
            x, out_image, _ = stage(input)
            
            out_images.append(out_image)
            if len(self.stages) > current_stage+1:
                x = self.ups[current_stage](x)
            input = (img, cond, x)
        return out_images

        


class MultiScaleLoss(nn.Module):
    def __init__(self, loss_func=nn.L1Loss()):
        super().__init__()
        self.loss_func=loss_func

    def forward(self, outputs, gt):
        loss = 0
        for output in outputs:
            scaled = nn.functional.interpolate(gt, size=output.shape[-2:])
            loss += self.loss_func(output, scaled)
        return loss