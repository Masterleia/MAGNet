import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # self.use_cuda = args.cuda
        self.P = args.seq_len
        self.m = args.enc_in
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pre = args.pred_len
        self.pt = int((self.P - self.Ck) / self.skip)
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m * self.pre)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m * self.pre)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, self.pre)
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)  # (128,1,168,8)
        c = F.relu(self.conv1(c))  # (128,50,163,1)
        c = self.dropout(c)  # (128,50,163,1)
        c = torch.squeeze(c, 3)  # (128,50,163)

        # RNN
        r = c.permute(2, 0, 1).contiguous() # (163,128,50)
        _, r = self.GRU1(r)  # (1,128,50)
        r = self.dropout(torch.squeeze(r, 0))
        # r = r.view(batch_size*self.m, -1)

        # skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            # s = s.view(batch_size*self.m, -1)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]  # (128,24,8)
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)  # (1024,24)
            z = self.highway(z)  # (1024,1)
            z = z.view(-1, self.m * self.pre)
            res = res + z

        res = res.view(batch_size, self.pre, self.m)

        if (self.output):
            res = self.output(res)
        return res




