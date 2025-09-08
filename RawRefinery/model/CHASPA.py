import torch.nn.functional as F
import torch
import torch.nn as nn

def interleaved_chunk(x):
    B, C, W, H = x.shape

    y = x.view(B, C // 2, 2, W, H)

    chunk1 = y[:, :, 0, :, :]
    chunk2 = y[:, :, 1, :, :]  
    return chunk1, chunk2

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)



class FastLayerNorm2d(nn.Module):
    """
    The fast and correct way to implement LayerNorm2d.
    This uses nn.GroupNorm with 1 group, which is mathematically
    equivalent and uses a highly optimized native implementation.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        # The learnable parameters (weight and bias) are handled inside GroupNorm.
        # They are of shape (channels,).
        self.gn = nn.GroupNorm(1, channels, eps=eps)

    def forward(self, x):
        return self.gn(x)
    
class RMSNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-8, affine=True):
        """
        Root Mean Square Normalization for 2D feature maps.
        Args:
            num_channels: number of channels (C)
            eps: numerical stability constant
            affine: whether to include learnable scale and bias per channel
        """
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # Compute RMS over channel dimension only
        rms = x.pow(2).mean(dim=1, keepdim=True).sqrt()
        x = x / (rms + self.eps)

        if self.affine:
            x = x * self.weight + self.bias
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ConditionedChannelAttention(nn.Module):
    def __init__(self, dims, cat_dims):
        super().__init__()
        in_dim = dims + cat_dims
        self.mlp = nn.Sequential(nn.Linear(in_dim, dims))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, conditioning):
        pool = self.pool(x)
        conditioning = conditioning.unsqueeze(-1).unsqueeze(-1)
        cat_channels = torch.cat([pool, conditioning], dim=1)
        cat_channels = cat_channels.permute(0, 2, 3, 1)
        ca = self.mlp(cat_channels).permute(0, 3, 1, 2)

        return ca 
    

class ConditionedChannelAttentionWrapper(nn.Module):
    def __init__(self, dims, cat_dims):
        super().__init__()
        self.CCAW = ConditionedChannelAttention(dims, cat_dims)
    def forward(self, input):
        inp = input[0]
        cond = input[1]
        x = self.CCAW(inp, cond)
        return (inp * (1 + x), cond)


class NKA(nn.Module):
    def __init__(self, dim, channel_reduction = 8):
        super().__init__()

        reduced_channels = dim // channel_reduction
        self.proj_1 = nn.Conv2d(dim, reduced_channels, 1, 1, 0)
        self.dwconv = nn.Conv2d(reduced_channels, reduced_channels, 3, 1, 1, groups=reduced_channels)
        self.proj_2 = nn.Conv2d(reduced_channels, reduced_channels * 2, 1, 1, 0)
        self.sg = SimpleGate()
        self.attention = nn.Conv2d(reduced_channels, dim, 1, 1, 0)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # First projection to a smaller dimension
        y = self.proj_1(x)
        # DW conv
        attn = self.dwconv(y)
        # PW back to orignal space
        attn = self.proj_2(attn)
        # Non-linearity
        attn = self.sg(attn)
        # Apply attention map
        out = x * self.attention(attn)
        return out

class LKA(nn.Module):
    def __init__(self, dim, channel_reduction = 8):
        super().__init__()

        reduced_channels = dim // channel_reduction
        self.proj_1 = nn.Conv2d(dim, reduced_channels, 1, 1, 0)
        self.dwconv = nn.Conv2d(reduced_channels, reduced_channels, 7, 1, 3, groups=reduced_channels)
        self.proj_2 = nn.Conv2d(reduced_channels, reduced_channels * 2, 1, 1, 0)
        self.sg = SimpleGate()
        self.attention = nn.Conv2d(reduced_channels, dim, 1, 1, 0)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # First projection to a smaller dimension
        y = self.proj_1(x)
        # DW conv
        attn = self.dwconv(y)
        # PW back to orignal space
        attn = self.proj_2(attn)
        # Non-linearity
        attn = self.sg(attn)
        # Apply attention map
        out = x * self.attention(attn)
        return out




class CHASPABlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0, cond_chans=0):
        super().__init__()
        dw_channel = c * DW_Expand

        self.NKA = NKA(c)
        self.conv1 =  nn.Conv2d(
            in_channels=c,
            out_channels=c,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=c,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = ConditionedChannelAttention(dw_channel // 2, cond_chans)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv2 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # self.grn = GRN(ffn_channel // 2)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, input):
        inp = input[0]
        cond = input[1]

        x = inp
        x = self.norm1(x)

        # Channel Mixing
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x, cond)
        x = self.conv3(x)
        x = self.dropout2(x)
        y = inp + x * self.beta

        #Spatial Mixing
        x = self.NKA(self.norm2(y))
        x = self.conv1(x)
        x = self.dropout1(x)
        

        return (y + x * self.gamma, cond)
    

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

class CHASPA(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=16,
        mid_blk_nums=[],
        enc_blk_nums=[],
        dec_blk_nums=[],
        cond_input=1,
        cond_output=32,
        expand_dims=2,
        drop_out_rate=0.0,
        drop_out_rate_increment=0.0
    ):
        super().__init__()

        self.expand_dims = expand_dims
        self.conditioning_gen = nn.Sequential(
            nn.Linear(cond_input, 64), nn.ReLU(), nn.Dropout(drop_out_rate), nn.Linear(64, cond_output),
        )

        self.intro = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.fusers = nn.ModuleList()
        self.middles = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        # Bottle Neck Blocks
        num = mid_blk_nums[0]
        self.middles.append(
            nn.Sequential(
                *[
                    CHASPABlock(chan, cond_chans=cond_output, drop_out_rate=drop_out_rate)
                    for _ in range(num)
                ]
            )
        )
        
        # for num in enc_blk_nums:
        for i in range(len(enc_blk_nums)):
            num = enc_blk_nums[i]
            self.encoders.append(
                nn.Sequential(ConditionedChannelAttentionWrapper(chan, cond_output),
                    *[
                        CHASPABlock(chan, cond_chans=cond_output, drop_out_rate=drop_out_rate)
                        for _ in range(num)
                    ]
                )
            )
            drop_out_rate += drop_out_rate_increment 
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
            # Bottle Neck Blocks
            num = mid_blk_nums[i+1]
            self.middles.append(
                nn.Sequential(
                    *[
                        CHASPABlock(chan, cond_chans=cond_output, drop_out_rate=drop_out_rate)
                        for _ in range(num)
                    ]
                )
            )
            
            


        for i in range(len(dec_blk_nums)):
            num = dec_blk_nums[::-1][i]
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            drop_out_rate -= drop_out_rate_increment 
            chan = chan // 2
            self.fusers.append(Fuser(chan))
            self.decoders.append(
                nn.Sequential(
                    *[
                        CHASPABlock(chan, cond_chans=cond_output, drop_out_rate=drop_out_rate)
                        for _ in range(num)
                    ]
                )
            )


        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, cond_in):
        # Conditioning:
        cond = self.conditioning_gen(cond_in)

        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        mids = []
        for encoder, down, middles in zip(self.encoders, self.downs, self.middles[:-1]):
            x = encoder((x, cond))[0]
            mids.append(middles((x, cond))[0])
            x = down(x)
        
        x = self.middles[-1]((x, cond))[0]

        for decoder, fuser, up, mid_skip in zip(self.decoders, self.fusers, self.ups, mids[::-1]):
            x = up(x)
            x = fuser(x, mid_skip)
            x = decoder((x, cond))[0]

        x = self.ending(x)
    
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

def make_atto(in_chan, out_chan, cond_chans):
    model = CHASPA(
        width=64,
        enc_blk_nums=[0, 1, 0],
        mid_blk_nums=[0, 0, 0, 3],
        dec_blk_nums=[0, 2, 1],
        cond_input=cond_chans,
        in_channels=in_chan,
        out_channels=out_chan,
    )
    return model


def make_conditioning(conditioning, alpha):
    B = conditioning.shape[0]
    conditioning_extended = torch.zeros(B, 2)
    conditioning_extended[:, 0] = conditioning
    conditioning_extended[:, 1] = alpha
    return conditioning_extended