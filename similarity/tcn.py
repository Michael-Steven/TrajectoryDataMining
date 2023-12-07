import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
# from torch.optim import Adam


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        # self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        # self.chomp2 = Chomp1d(padding)
        # self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        # # self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.bn2, self.relu2, self.dropout2)
        self.down_sample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        # self.conv1.weight.data.normal_ (0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        if self.down_sample is not None:
            self.down_sample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.down_sample is None else self.down_sample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, seq_len):
        x = self.network(x)#.transpose(1, 2)
        x = self.pooling(x).squeeze(2)
        # x = x[torch.arange(seq_len.shape[0]), seq_len - 1]
        return x


# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
#         super(TCN, self).__init__()
#         self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
#         # self.linear = nn.Linear(num_channels[-1], output_size)
#         # self.pooling = nn.AdaptiveMaxPool1d(1)
#         # self.init_weights()

#     def init_weights(self):
#         self.linear.weight.data.normal_(0, 0.01)

#     def forward(self, x, seq_len):
#         y1 = self.tcn(x).transpose(1, 2)
#         y1 = y1[torch.arange(seq_len.shape[0]), seq_len - 1]
#         # y1 = self.linear(y1)
#         return y1


# tcn = TemporalConvNet(num_inputs=128, num_channels=[32]*2+[128], kernel_size=3, dropout=0)
# # lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False, num_layers=1)
# # attn_encoder = nn.MultiheadAttention(embed_dim=128, num_heads=8)
# test_input = torch.rand((256, 100, 128))
# seqlen = torch.randint(5, 100, (256, ))
# for i in range(1):
#     test_output = tcn(test_input.transpose(1, 2), seqlen)
#     # test_output, _ = lstm(test_input)
#     # test_output = test_output[torch.arange(256), 100 - 1]
# print(test_output.shape)
