import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        dim = 64
        self.conv2d_features3 = nn.Sequential(
            nn.Conv2d(dim*2, out_channels=dim*2, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(),
        )

        self.norm1 = nn.BatchNorm2d(dim * 2)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        out1 = self.conv2d_features3(out)
        out = out1 + out
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        dim = 64
        self.conv2d_features3 = nn.Sequential(
            nn.Conv2d(dim*2, out_channels=dim*2, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(),
        )
        self.norm1 = nn.BatchNorm2d(dim*2)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        out1 = self.conv2d_features3(out)
        out = out1 + out
        return out


class AGA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=8, dropout=0.1, agent_num=49):
        super(AGA, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.agent_num = agent_num

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(inner_dim, dim, bias=False)
        self.to_v = nn.Linear(inner_dim, dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim*2, dim*2),
            nn.Dropout(dropout)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(49, dim))

    def forward(self, x1, x2, kv_include_self=False):
        b, n, _, h = *x1.shape, self.heads
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x2)

        agent_tokens = self.pool(q)
        agent_tokens = rearrange(agent_tokens, 'b n (h d) -> b h n d', h=h)
        q1 = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v1 = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dots0 = einsum('b h i d, b h j d -> b h i j', agent_tokens, k) * self.scale
        attn0 = self.attend(dots0)
        attn0 = self.dropout(attn0)
        out0 = einsum('b h i j, b h j d -> b h i d', attn0, v1)

        dots1 = einsum('b h i d, b h j d -> b h i j', q1, agent_tokens) * self.scale
        attn1 = self.attend(dots1)
        attn1 = self.dropout(attn1)
        out1 = einsum('b h i j, b h j d -> b h i d', attn1, out0)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out1 = torch.cat((out1, x1), dim=-1)
        out = self.to_out(out1)

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension hidden_features and out_features
        output: (B, N, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


BATCH_SIZE_TRAIN = 1
NUM_CLASS = 8


class MyTrans(nn.Module):
    def __init__(
            self,
            in_channelsd=4,
            num_classes=NUM_CLASS,
            num_tokens=4,
            dim=64,
            in_chansb=10,
            in_chans=14,
    ):
        super().__init__()
        self.L = num_tokens
        self.cT = dim
        new_bands = in_chans
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=in_chansb, out_channels=dim, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
        )

        self.conv2d_features2 = nn.Sequential(
            nn.Conv2d(in_channelsd, out_channels=dim, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        self.spe = CAM_Module(dim*2)
        self.spa = PAM_Module(dim*2)
        self.cross1 = AGA(dim)
        self.cross2 = AGA(dim)

        self.Mlp = Mlp(dim*2, dim*2, dim*2)
        self.fc1 = nn.Linear(dim*2, num_classes)
        self.normd = nn.LayerNorm(num_classes)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b0 = 10
        d0 = 4

        x = self.pad(x).squeeze(dim=1)

        B, N, d, d = x.shape
        b = torch.ones((B, b0, d, d))
        d = torch.ones((B, d0, d, d))
        b = b.to(device)
        b[:, :, :, :] = x[:, 0:b0, :, :]
        d[:, :, :, :] = x[:, b0:b0 + d0, :, :]
        b = b.to(device)
        d = d.to(device)

        b0 = b.unsqueeze(dim=-1)
        x1 = self.conv3d_features(b0)
        x1 = rearrange(x1, 'b c h w y ->b (h w) (c y)')

        x2 = self.conv2d_features2(d)
        x2 = x2.flatten(2).transpose(-1, -2)

        x_1 = self.cross1(x1, x2)
        x_2 = self.cross2(x2, x1)  # 这个模块加不加精度差不多

        x = x_1+x_2
        B, N, C = x.shape
        x = x.reshape(B, 7, 7, C).permute(0, 3, 1, 2)
        x_1 = self.spe(x)
        x_2 = self.spa(x)
        x = x_1+x_2
        x = x.flatten(2).transpose(-1, -2)

        x = self.Mlp(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.normd(x)

        return x


def NEWdlst(dataset, patch_size):
    model = None
    if dataset == 'Ori':
        model = MyTrans(in_chans=14, num_classes=8)
    return model