import torch.nn.functional as F
import torch
import torch.nn as nn

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


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ConditionedChannelAttention(nn.Module):
    def __init__(self, dims, cat_dims):
        super().__init__()
        in_dim = dims + cat_dims
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_dim, int(in_dim*1.5)),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(int(in_dim*1.5), dims)
        # )
        self.mlp = nn.Sequential(nn.Linear(in_dim, dims))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, conditioning):
        pool = self.pool(x)
        conditioning = conditioning.unsqueeze(-1).unsqueeze(-1)
        cat_channels = torch.cat([pool, conditioning], dim=1)
        cat_channels = cat_channels.permute(0, 2, 3, 1)
        ca = self.mlp(cat_channels).permute(0, 3, 1, 2)

        return ca
    

class NAFBlock0(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0, cond_chans=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = ConditionedChannelAttention(dw_channel // 2, cond_chans)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
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

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x, cond)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        # Channel Mixing
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        # x = self.grn(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return (y + x * self.gamma, cond)


class CondSEBlock(nn.Module):
    def __init__(self, chan, reduction=16, cond_chan=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(chan + cond_chan, chan // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(chan // reduction, chan, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, conditioning):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.cat([y, conditioning], dim=1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class CondFuser(nn.Module):
    def __init__(self, chan, cond_chan=1):
        super().__init__()
        self.cca = ConditionedChannelAttention(chan * 2, cond_chan)

    def forward(self, x1, x2, cond):
        x = torch.cat([x1, x2], dim=1)
        x = self.cca(x, cond) * x
        x1, x2 = x.chunk(2, dim=1)
        return x1 + x2
    
    
class Restorer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        chans = [],
        cond_input=1,
        cond_output=32,
        expand_dims=2,
        drop_out_rate=0.0,
        drop_out_rate_increment=0.0
    ):
        super().__init__()
        width = chans[0]

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
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.merges = nn.ModuleList()


        # for num in enc_blk_nums:
        for i in range(len(enc_blk_nums)):
            current_chan = chans[i]
            next_chan = chans[i + 1]
            num = enc_blk_nums[i]
            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock0(current_chan, cond_chans=cond_output, drop_out_rate=drop_out_rate)
                        for _ in range(num)
                    ]
                )
            )
            drop_out_rate += drop_out_rate_increment 
            self.downs.append(nn.Conv2d(current_chan, next_chan, 2, 2))

        self.middle_blks = nn.Sequential(
            *[
                NAFBlock0(next_chan, cond_chans=cond_output, drop_out_rate=drop_out_rate)
                for _ in range(middle_blk_num)
            ]
        )

        for i in range(len(dec_blk_nums)):
            current_chan = chans[-i-1]
            next_chan = chans[-i-2]
            num = dec_blk_nums[i]
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(current_chan, next_chan * 2 ** 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            drop_out_rate -= drop_out_rate_increment 
            self.merges.append(CondFuser(next_chan, cond_chan=cond_output))
            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock0(next_chan, cond_chans=cond_output, drop_out_rate=drop_out_rate)
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

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder((x, cond))[0]
            encs.append(x)
            x = down(x)

        x = self.middle_blks((x, cond))[0]

        for decoder, up, merge, enc_skip in zip(self.decoders, self.ups, self.merges, encs[::-1]):
            x = up(x)
            x = merge(x, enc_skip, cond)
            x = decoder((x, cond))[0]

        x = self.ending(x)

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    

class ModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Restorer(
            chans = [32, 128, 256, 512, 1024],
            enc_blk_nums = [1,1,3,4],
            middle_blk_num = 6,
            dec_blk_nums = [2, 2, 1, 1],
            cond_input = 1,
            in_channels = 3,
            out_channels = 3, 
        )

    def forward(self, x, cond, residual):
        output = self.model(x, cond)
        return residual + output
    
def make_sparse(model_name = '/Volumes/EasyStore/models/Cond_NAF_variable_layers_cca_merge_unet_sparse_ssim_real_raw.pt'):
    model = ModelWrapper()
    if model_name is not None:
        state_dict = torch.load(model_name, map_location="cpu")
        model.load_state_dict(state_dict)
    return model
