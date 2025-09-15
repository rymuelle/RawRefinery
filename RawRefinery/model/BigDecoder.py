from RawRefinery.model.Cond_NAFNet import ConditionedChannelAttention, SimpleGate, NAFBlock0
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
    
class Fuser(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.pwconv = nn.Conv2d(chan * 2, chan * 2, 1, 1, 0)
        #self.NKA = NKA(chan * 2)
        self.sg = SimpleGate()
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.pwconv(x)
        #x = self.NKA(x)
        x = self.sg(x)
        return x
    
class ScaleAndProcess(nn.Module):
    def __init__(self, in_chan, width, out_chan, n_blocks, scale, cond_chans=1, ):
        super().__init__()
        assert np.log(scale)/np.log(2) == int(np.log(scale)/np.log(2)), f"Scale ({scale}) must be a multiple of 2"
        padding = max(scale // 2 - 1, 0)
        self.scale = scale
        self.in_conv = nn.Conv2d(in_chan, width, scale, scale, padding)
        self.blocks = nn.Sequential(
            *[NAFBlock0(width, cond_chans=cond_chans) for _ in range(n_blocks)]
        )
        self.fuser = Fuser(width)
        self.out_conv = nn.Conv2d(width, out_chan, 1, 1, 0)
    
    def forward(self, input):
        if len(input) == 2:
            in_image, cond = input
        elif len(input)==3:
            in_image, cond, previous_output = input
        after_conv = self.in_conv(in_image)
        if len(input) == 3:
            after_conv = self.fuser(after_conv, previous_output)
            
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
        return loss/len(outputs)
    



##
## Hierarchical
##

class StepDownBlock(nn.Module):
    def __init__(self, chan, scale, block, n_block, cond_chans=1):
        super().__init__()
        padding = max(scale // 2 - 1, 0)
        self.down = nn.Conv2d(chan, chan, scale, scale, padding)
        self.block = nn.Sequential(*[block(chan, cond_chans=cond_chans) for _ in range(n_block)])
        self.up = nn.Sequential(nn.Conv2d(chan, chan * scale ** 2, 1, 1, 0), nn.PixelShuffle(scale))
        self.beta = nn.Parameter(torch.zeros(1, chan, 1, 1), requires_grad=True)
    def forward(self, input):
        x = input[0]
        cond = input[1]
        x = self.down(x)
        x = self.block((x, cond))[0]
        x = self.up(x)
        # Residual
        x = self.beta * x + input[0]
        return (x, cond)
    

class HierarchicalBlock(nn.Module):
    def __init__(self, chan, scales=[8, 4, 2, 1], n_blocks=[1, 1, 1, 1], block=NAFBlock0, cond_chans=1):
        super().__init__()
        self.stages = nn.ModuleList()
        for scale, n_block in zip(scales, n_blocks):
            if scale!=1:
                padding = max(scale // 2 - 1, 0)
                self.stages.append(StepDownBlock(chan, scale, block, n_block, cond_chans=cond_chans))
            else:
                self.stages.append(block(chan, cond_chans=cond_chans))
    
    def forward(self, input):
        x = input[0]
        cond = input[1]
        
        for stage in self.stages:
            x, _ = stage( (x, cond))
            
        return (x, cond)


class HBlockModel(nn.Module):
    def __init__(self, chan, width, n_hblocks, scales=[8, 4, 2, 1],  n_blocks=[2, 1, 1, 1], block=NAFBlock0):
        super().__init__()
        self.in_conv = nn.Conv2d(chan, width, 3, 1, 1)
        self.blocks = nn.ModuleList()
        for i in range(n_hblocks):
            self.blocks.append(HierarchicalBlock(width, scales=scales,  n_blocks=n_blocks, block=block))
            self.out_convs = nn.Conv2d(width, chan, 3, 1, 1)
 
    def forward(self, x, cond):
        x = self.in_conv(x)
        inp = x
        outputs = []
        for block in self.blocks:
            x = block( (x, cond) )[0]
            # Addind detail back in with residual connection
            # x = x + inp
            outputs.append(self.out_convs(x))
        return outputs
    

class SlapTwoBigDecoders(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.decoder1 = BigDecoder(*args, **kwargs)
        self.decoder2 = BigDecoder(*args, **kwargs)
    def forward(self, input):
        x = input[0]
        cond = input[1]
        out1 = self.decoder1( x, cond)
        out2 = self.decoder2( out1[-1], cond)
        return out1 + out2
    


class BigStep2(nn.Module):
    def __init__(self, in_chan, base_chan, 
                 block_type = NAFBlock0,  
                 n_blocks = [1, 1, 1, 1, 1],
                 cond_chans=1):
        super().__init__()
        # self.intro_conv = nn.Conv2d(in_chan, base_chan, 3, 1, 1)

        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.merges = nn.ModuleList()
        chan = base_chan
        for scale_power, n_block  in enumerate(n_blocks):
            scale = 2 ** scale_power
            padding = max(scale // 2 - 1, 0)
            self.downs.append(nn.Conv2d(in_chan, chan, scale, scale, padding))
            block_list = [NAFBlock0(chan, cond_chans=cond_chans) for _ in range(n_block)]
            self.blocks.append(nn.Sequential(
                *block_list
            ))

            if scale > 1:
                up = nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            else:
                up = nn.Identity()

            self.ups.append(up)

            fuser = Fuser(chan)
            self.merges.append(fuser)

            chan = chan * 2

        self.out_conv = nn.Conv2d(base_chan, in_chan, 3, 1, 1)
        self.downs = self.downs[::-1]
        self.blocks = self.blocks[::-1]
        self.ups = self.ups[::-1]
        self.merges = self.merges[::-1]
    def forward(self, image, cond):
        stage = 0
        for down, block, up, merge in zip(self.downs, self.blocks, self.ups, self.merges):
            x = down(image)
            if stage:
                x = merge(x, up_x)
            x = block((x, cond))[0]
            up_x = up(x)
            stage += 1
        start = perf_counter()
        out = self.out_conv(up_x)
        return [image + out]
