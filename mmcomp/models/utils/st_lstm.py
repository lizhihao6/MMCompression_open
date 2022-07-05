import torch
import torch.nn as nn


class CA_LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bias = bias
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv1 = nn.Conv2d(in_channels=input_dim + 2 * hidden_dim, out_channels=3 * hidden_dim,
                               kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=input_dim + 2 * hidden_dim, out_channels=3 * hidden_dim,
                               kernel_size=kernel_size, padding=padding, bias=bias)

        self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                               kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv4 = nn.Conv2d(in_channels=input_dim + 2 * hidden_dim, out_channels=hidden_dim,
                               kernel_size=kernel_size, padding=padding, bias=self.bias)
        self.conv5 = nn.Conv2d(in_channels=2 * hidden_dim, out_channels=hidden_dim,
                               kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, x, h, c, M):
        '''
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self.init_hidden(batch_size=input_tensor.size(0))
        '''
        b, _, height, width = x.shape
        if h is None:
            h = torch.zeros([b, self.hidden_dim, height, width]).to(x.device)
        if c is None:
            c = torch.zeros([b, self.hidden_dim, height, width]).to(x.device)
        if M is None:
            M = torch.zeros([b, self.hidden_dim, height, width]).to(x.device)

        combined1 = torch.cat([x, h, c], dim=1)  # concatenate along channel axis
        combined1_conv = self.conv1(combined1)
        cc_i, cc_f, cc_g = torch.split(combined1_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        c = f * c + i * g

        combined2 = torch.cat([x, c, M], dim=1)
        combined2_conv = self.conv2(combined2)
        cc_i_, cc_f_, cc_g_ = torch.split(combined2_conv, self.hidden_dim, dim=1)
        i_ = torch.sigmoid(cc_i_)
        f_ = torch.sigmoid(cc_f_)
        g_ = torch.tanh(cc_g_)
        M_ = torch.tanh(self.conv3(M))
        M_ = f_ * M_ + i_ * g_

        combined4 = torch.cat([x, c, M_], dim=1)
        combined4_conv = self.conv4(combined4)
        o = torch.tanh(combined4_conv)
        combined5 = torch.cat([c, M_], dim=1)
        h = o * torch.tanh(self.conv5(combined5))

        return h, c, M_


class Prior_STLSTM(nn.Module):
    def __init__(self, concate_channel, hidden_channel):
        super().__init__()
        self.hidden_channel = hidden_channel
        self.ca_lstm1 = CA_LSTM(concate_channel, hidden_channel, (3, 3), False)
        self.ca_lstm2 = CA_LSTM(hidden_channel, hidden_channel, (3, 3), False)

    def forward(self, combined_feat, h=None, c=None, M=None):
        # todo check here
        if h is None or c is None:
            h0, h1, c0, c1 = None, None, None, None
        else:
            h0, h1 = torch.split(h, self.hidden_channel, dim=1)
            c0, c1 = torch.split(c, self.hidden_channel, dim=1)

        _h0, _c0, _M = self.ca_lstm1(combined_feat, h0, c0, M)
        _h1, _c1, _M = self.ca_lstm2(_h0, h1, c1, _M)
        h = torch.cat([_h0, _h1], dim=1)
        c = torch.cat([_c0, _c1], dim=1)
        M = _M
        out = combined_feat + _h1
        return out, h, c, M
