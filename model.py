import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence



class _NonLocalBlockND(nn.Module):
    """PyTorch module for NonLocalBlock"""

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
    """PyTorch module for NonLocalBlock2D"""
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


# CONVLSTM: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
class ConvLSTMCell(nn.Module):
    """PyTorch module for ConvLSTM Cell"""
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
    """PyTorch module for ConvLSTM"""

    def __init__(self, input_dim, hidden_dim):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim=self.input_dim, hidden_dim=self.hidden_dim)

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
    """PyTorch module for Convolutional layer+Batch Normalization with ReLU"""

    def __init__(self,ins,ous,kernel,padding = 0):
        super(Convbn,self).__init__()

        self.conv = nn.Conv2d(ins,ous,kernel,padding = padding)
        self.bn = nn.BatchNorm2d(ous)
        self.relu = nn.ReLU(inplace = True)

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class TemporalAttension(nn.Module):
    """PyTorch module for RealSmileNet's TSA Block"""
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
    """PyTorch module for RealSmileNet and AUDA features classification"""

    def __init__(self, f):
        super(DeepSmileNet,self).__init__()
        concat_size_on_last = 0
        self.f = f

        if "videos" in f:
            self.TSA = TemporalAttension(3)
            self.FPN = self._fpn_layers([4,'M', 6, 'M'])
            self.ConvLSTMLayer = ConvLSTM(6,8)
            self.Classification = nn.Sequential(
                NONLocalBlock2D(8),
                nn.AvgPool2d(kernel_size = 2, stride = 2),
                Convbn(8,10,2),
                nn.Dropout(0.5),
                nn.Flatten()  # 250
            )
            concat_size_on_last += 250

        if "aus" in f:
            n = 150
            self.AUsLSTM = nn.LSTM(input_size=17, hidden_size=n, num_layers=1, batch_first=True)
            self.ClassificationAUs = nn.Sequential(
                nn.BatchNorm1d(n),
                nn.ReLU(inplace=True)
            )
            concat_size_on_last += n

        self.Ddict = {}

        if "d1da27" in f:
            n = 150
            self.Fd1da27LSTM = nn.LSTM(input_size=17, hidden_size=n, num_layers=1, batch_first=True)
            self.ClassificationFd1da27 = nn.Sequential(
                nn.BatchNorm1d(n),
                nn.ReLU(inplace=True)
            )
            self.Ddict.update({"d1da27": ("dynamics_delta_adjusted_27", self.Fd1da27LSTM, self.ClassificationFd1da27)})
            concat_size_on_last += n

        if "d2da27" in f:
            n = 150
            self.Fd2da27LSTM = nn.LSTM(input_size=17, hidden_size=n, num_layers=1, batch_first=True)
            self.ClassificationFd2da27 = nn.Sequential(
                nn.BatchNorm1d(n),
                nn.ReLU(inplace=True)
            )
            self.Ddict.update({"d2da27": ("dynamics_2nd_delta_adjusted_27", self.Fd2da27LSTM, self.ClassificationFd2da27)})
            concat_size_on_last += n

        if "d1da9" in f:
            n = 150
            self.Fd1da9LSTM = nn.LSTM(input_size=17, hidden_size=n, num_layers=1, batch_first=True)
            self.ClassificationFd1da9 = nn.Sequential(
                nn.BatchNorm1d(n),
                nn.ReLU(inplace=True)
            )
            self.Ddict.update({"d1da9": ("dynamics_delta_adjusted_9", self.Fd1da9LSTM, self.ClassificationFd1da9)})
            concat_size_on_last += n

        if "d2da9" in f:
            n = 150
            self.Fd2da9LSTM = nn.LSTM(input_size=17, hidden_size=n, num_layers=1, batch_first=True)
            self.ClassificationFd2da9 = nn.Sequential(
                nn.BatchNorm1d(n),
                nn.ReLU(inplace=True)
            )
            self.Ddict.update({"d2da9": ("dynamics_2nd_delta_adjusted_9", self.Fd2da9LSTM, self.ClassificationFd2da9)})
            concat_size_on_last += n

        if "d1d27" in f:
            n = 150
            self.Fd1d27LSTM = nn.LSTM(input_size=17, hidden_size=n, num_layers=1, batch_first=True)
            self.ClassificationFd1d27 = nn.Sequential(
                nn.BatchNorm1d(n),
                nn.ReLU(inplace=True)
            )
            self.Ddict.update({"d1d27": ("dynamics_delta_27", self.Fd1d27LSTM, self.ClassificationFd1d27)})
            concat_size_on_last += n

        if "d2d27" in f:
            n = 150
            self.Fd2d27LSTM = nn.LSTM(input_size=17, hidden_size=n, num_layers=1, batch_first=True)
            self.ClassificationFd2d27 = nn.Sequential(
                nn.BatchNorm1d(n),
                nn.ReLU(inplace=True)
            )
            self.Ddict.update({"d2d27": ("dynamics_2nd_delta_27", self.Fd2d27LSTM, self.ClassificationFd2d27)})
            concat_size_on_last += n

        if "d1d9" in f:
            n = 150
            self.Fd1d9LSTM = nn.LSTM(input_size=17, hidden_size=n, num_layers=1, batch_first=True)
            self.ClassificationFd1d9 = nn.Sequential(
                nn.BatchNorm1d(n),
                nn.ReLU(inplace=True)
            )
            self.Ddict.update({"d1d9": ("dynamics_delta_9", self.Fd1d9LSTM, self.ClassificationFd1d9)})
            concat_size_on_last += n

        if "d2d9" in f:
            n = 150
            self.Fd2d9LSTM = nn.LSTM(input_size=17, hidden_size=n, num_layers=1, batch_first=True)
            self.ClassificationFd2d9 = nn.Sequential(
                nn.BatchNorm1d(n),
                nn.ReLU(inplace=True)
            )
            self.Ddict.update({"d2d9": ("dynamics_2nd_delta_9", self.Fd2d9LSTM, self.ClassificationFd2d9)})
            concat_size_on_last += n

        if "si" in f:
            n = 10
            self.SILSTM = nn.LSTM(input_size=1, hidden_size=n, num_layers=1, batch_first=True)
            self.ClassificationSI = nn.Sequential(
                nn.BatchNorm1d(n),
                nn.ReLU(inplace=True)
            )
            concat_size_on_last += n

        if "auwise" in f:
            n = 80
            npart = int(n/5)
            self.AUWISEsimple_apex = nn.Sequential(
                nn.BatchNorm1d(119),
                nn.Linear(119, npart),
                nn.ReLU(inplace = True)
            )

            self.AUWISEsimple_offset = nn.Sequential(
                nn.BatchNorm1d(119),
                nn.Linear(119, npart),
                nn.ReLU(inplace = True)
            )

            self.AUWISEsimple_onset = nn.Sequential(
                nn.BatchNorm1d(119),
                nn.Linear(119, npart),
                nn.ReLU(inplace = True)
            )

            self.AUWISEsimple_whole_sequence = nn.Sequential(
                nn.BatchNorm1d(119),
                nn.Linear(119, npart),
                nn.ReLU(inplace = True)
            )

            self.AUWISEsimple_whole_smile = nn.Sequential(
                nn.BatchNorm1d(119),
                nn.Linear(119, npart),
                nn.ReLU(inplace = True)
            )

            self.AUWISEdict = [
                self.AUWISEsimple_apex,
                self.AUWISEsimple_offset,
                self.AUWISEsimple_onset,
                self.AUWISEsimple_whole_sequence,
                self.AUWISEsimple_whole_smile
            ]
            concat_size_on_last += n

        if "crossau" in f:
            m = 640
            n = 80
            npart = int(n / 5)
            mpart = int(m / 5)

            self.CROSSAUsimple_apex = nn.Sequential(
                nn.BatchNorm1d(1088),
                nn.Linear(1088, mpart),
                nn.ReLU(inplace=True),
                nn.Linear(mpart, npart),
                nn.ReLU(inplace=True)
            )

            self.CROSSAUsimple_offset = nn.Sequential(
                nn.BatchNorm1d(1088),
                nn.Linear(1088, mpart),
                nn.ReLU(inplace=True),
                nn.Linear(mpart, npart),
                nn.ReLU(inplace=True)
            )

            self.CROSSAUsimple_onset = nn.Sequential(
                nn.BatchNorm1d(1088),
                nn.Linear(1088, mpart),
                nn.ReLU(inplace=True),
                nn.Linear(mpart, npart),
                nn.ReLU(inplace=True)
            )

            self.CROSSAUsimple_whole_sequence = nn.Sequential(
                nn.BatchNorm1d(1088),
                nn.Linear(1088, mpart),
                nn.ReLU(inplace=True),
                nn.Linear(mpart, npart),
                nn.ReLU(inplace=True)
            )

            self.CROSSAUsimple_whole_smile = nn.Sequential(
                nn.BatchNorm1d(1088),
                nn.Linear(1088, mpart),
                nn.ReLU(inplace=True),
                nn.Linear(mpart, npart),
                nn.ReLU(inplace=True)
            )

            self.CROSSAUdict = [
                self.CROSSAUsimple_apex,
                self.CROSSAUsimple_offset,
                self.CROSSAUsimple_onset,
                self.CROSSAUsimple_whole_sequence,
                self.CROSSAUsimple_whole_smile
            ]
            concat_size_on_last += n

        self.ClassificationCat = nn.Sequential(
            nn.Linear(concat_size_on_last, 1),
            nn.Sigmoid()
        )


    def _fpn_layers(self,cfg,in_channels = 3):
        """Creates module for RealSmileNet's FPN Block"""
        layers = [nn.BatchNorm2d(in_channels)]
        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Convbn(in_channels, x, kernel=3, padding=1)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.Dropout2d(0.2)]
        return nn.Sequential(*layers)

    def __forward_au_features(self, aus, aus_len, lstm_layer, cls_layer):
        """Forwards AUDA Package's sequential features"""

        # https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
        # https://github.com/pytorch/pytorch/issues/43227
        # RuntimeError: 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor - naprawione - aus_len z cpu podawane
        aus_packed = pack_padded_sequence(aus, aus_len, batch_first=True, enforce_sorted=False)
        aus_packed, (aus_hn, aus_cn) = lstm_layer(aus_packed)
        aus, _ = pad_packed_sequence(aus_packed, batch_first=True)

        # to samo
        # aus01 = aus[0][aus_len[0]-1]
        # aus02 = aus_hn[0][0]
        aus = aus_hn[0]
        aus = cls_layer(aus)
        return aus

    def forward(self, x_videos, s, x_df_dict, frames_len):
        # TSA block
        tocat = []

        if "videos" in self.f:
            x_videos = self.TSA(x_videos)

            # FPN block
            batch_size, timesteps, C, H, W = x_videos.size()
            input_x = []

            for l in range(x_videos.size(0)): # Pozbywanie się pustych klatek
                input_x.append(x_videos[l,s[l]:,:,:,:])
            #Przykładow: MAX S
            input_x = torch.cat(input_x,0)
            out = self.FPN(input_x)

            # ConvLSTM + LSTM blocks
            current = 0
            _,new_c,new_w,new_h = out.size()
            reshape_out = torch.zeros((batch_size, timesteps,new_c,new_w,new_h),device = x_videos.device)

            for index,l in enumerate(s):
                reshape_out[index,l:] = out[current:current + timesteps - l]
                current+= timesteps - l
            x_videos = reshape_out

            x_videos = self.ConvLSTMLayer(x_videos, s)
            x_videos = self.Classification(x_videos)
            tocat.append(x_videos)

        if "aus" in self.f:
            aus = self.__forward_au_features(x_df_dict['action_units'], frames_len, self.AUsLSTM, self.ClassificationAUs)
            tocat.append(aus)

        if "si" in self.f:
            si = self.__forward_au_features(x_df_dict['smile_intensities'], frames_len, self.SILSTM, self.ClassificationSI)
            tocat.append(si)

        d1d2features = [d1d2f for d1d2f in self.f if d1d2f.startswith("d1") or d1d2f.startswith("d2")]
        for d1d2feature in d1d2features:
            x_d1d2 = x_df_dict[self.Ddict.get(d1d2feature)[0]]
            d1d2lstm = self.Ddict.get(d1d2feature)[1]
            d1d2endclassifier = self.Ddict.get(d1d2feature)[2]
            aus = self.__forward_au_features(x_d1d2, frames_len, d1d2lstm, d1d2endclassifier)
            tocat.append(aus)

        if "auwise" in self.f:
            i = 0
            for value in self.AUWISEdict:
                x_auwise = x_df_dict['auwise'][:,i,:]
                x_auwise = value(x_auwise)
                tocat.append(x_auwise)
                i+=1

        if "crossau" in self.f:
            i = 0
            for value in self.CROSSAUdict:
                x_crossau = x_df_dict['crossau'][:,i,:]
                x_crossau = value(x_crossau)
                tocat.append(x_crossau)
                i+=1

        all_features = torch.cat(tocat,dim=1) #dim=1 - dim0 to u nas batch size

        all_features_to_sigm = self.ClassificationCat(all_features)
        return all_features_to_sigm, all_features

class MultipleDeepSmileNet(nn.Module):
    """PyTorch module for multiple DeepSmileNet modules (aka. concatenation models)"""
    def __init__(self, deepSmileNets, variant):
        self.variant = variant
        super(MultipleDeepSmileNet,self).__init__()

        n = 0

        if variant == 1:
            n = len(deepSmileNets)

            for submodel in deepSmileNets:
                for name, param in submodel.named_parameters():
                    param.requires_grad = False
        elif variant == 2:
            for submodel in deepSmileNets:
                for name, param in submodel.named_parameters():
                    param.requires_grad = False
                    if "ClassificationCat" in name and "weight" in name:
                        size = param.size()[1]
                        n+=size

        self.deepSmileNets = nn.ModuleList(deepSmileNets)
        self.Classification = nn.Sequential(
            nn.Linear(n, 1),
            nn.Sigmoid()
        )

        if variant == 1:
            linear = self.Classification[0]
            linear.weight.data.fill_(1/n)
            nn.init.constant_(linear.bias.data, 0)

    def forward(self, x_videos, s, x_df_dict, frames_len):
        preds = []
        for deepSmileNet in self.deepSmileNets:
            deepSmileNet.eval()
            pred, pred_pre_class = deepSmileNet(x_videos, s, x_df_dict, frames_len)

            if self.variant == 1:
                preds.append(pred)
            elif self.variant == 2:
                preds.append(pred_pre_class)
        cat_all = torch.cat(preds, dim=1)

        cat_all = self.Classification(cat_all)

        return cat_all, None