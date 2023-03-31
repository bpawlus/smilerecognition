import torch
from torch import nn

# NonLocalBlock: https://github.com/AlexHex7/Non-local_pytorch/tree/master/lib
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


# CONVLSTM: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=3,
                              padding=1)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell  = ConvLSTMCell(input_dim=self.input_dim, hidden_dim=self.hidden_dim)

    def forward(self, input_tensor,time = None):
        b, _, _, h, w = input_tensor.size()

        hidden_state = self.cell.init_hidden(b, h, w)
        seq_len = input_tensor.size(1)

        h, c = hidden_state
        for t in range(seq_len):
            reset = (time == t).nonzero().view(-1)
            for index in reset:
                h[index] = 0
                c[index] = 0

            h, c = self.cell(input_tensor=input_tensor[:, t, :, :, :],cur_state=[h, c])

        return h


#Conv2d + BN + RELU
class Convbn(nn.Module):
    def __init__(self,ins,ous,kernel,padding = 0):
        super(Convbn,self).__init__()

        self.conv = nn.Conv2d(ins,ous,kernel,padding = padding)
        self.bn = nn.BatchNorm2d(ous)
        self.relu = nn.ReLU(inplace = True)

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class TemporalAttension(nn.Module):
    def __init__(self,channels):
        super(TemporalAttension,self).__init__()

        self.conv = nn.Conv2d(channels,channels,3,padding = 1)

        init = torch.zeros((3,3))
        init[1,1] = 1

        self.conv.weight.data.copy_(init)
        self.conv.bias.data.copy_(torch.zeros(channels))

    def forward(self, x):
        x1 = x[:,:-1]
        x2 = x[:,1:]
        o = x2 - x1

        o = torch.cat((torch.zeros((x.size(0),1,x.size(2),x.size(3),x.size(4)),device = x.device),o),1)
        o = o.view(-1,x.size(2),x.size(3),x.size(4))
        x = self.conv(o).view(x.size()) * x + x

        return x


class DeepSmileNet(nn.Module):
    def __init__(self):
        super(DeepSmileNet,self).__init__()

        self.TSA = TemporalAttension(3)
        self.FPN = self._make_layers([4,'M', 6, 'M'])
        self.ConvLSTMLayer = ConvLSTM(6,8)
        self.Classification = nn.Sequential(
            NONLocalBlock2D(8),
            nn.AvgPool2d(kernel_size = 2, stride = 2),
            Convbn(8,10,2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(250,1),
            nn.Sigmoid()
        )

    def _make_layers(self,cfg,in_channels = 3):
        layers = [nn.BatchNorm2d(in_channels)]
        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Convbn(in_channels, x, kernel = 3, padding=1)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.Dropout2d(0.2)]
        return nn.Sequential(*layers)

    def _make_layers_org(self):
        layers = []

        layers += [Convbn(3, 4, kernel = 3, padding=1)]
        layers += [nn.BatchNorm2d(4)]
        layers += [nn.ReLU()]
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]

        layers += [Convbn(4, 8, kernel = 3, padding=1)]
        layers += [nn.BatchNorm2d(8)]
        layers += [nn.ReLU()]
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def forward(self,x,s):
        # TSA block
        x = self.TSA(x)

        # FPN block
        batch_size, timesteps, C, H, W = x.size()
        input_x = []

        for l in range(x.size(0)): # Pozbywanie się pustych klatek
            input_x.append(x[l,s[l]:,:,:,:])

        input_x = torch.cat(input_x,0)
        out = self.FPN(input_x)

        # ConvLTSM block
        current = 0
        _,new_c,new_w,new_h = out.size()
        reshape_out = torch.zeros((batch_size, timesteps,new_c,new_w,new_h),device = x.device)

        for index,l in enumerate(s):
            reshape_out[index,l:] = out[current:current + timesteps - l]
            current+= timesteps - l
        x  = reshape_out

        x = self.ConvLSTMLayer(x,s)

        # Classification block
        x = self.Classification(x)

        return x