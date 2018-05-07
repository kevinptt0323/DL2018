import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ShowAttendTellCell(nn.Module):
    def __init__(self, opt):
        super(ShowAttendTellCell, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        # self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size,
                self.rnn_size, bias=False, dropout=self.drop_prob_lm)

        if self.att_hid_size > 0:
            self.linear_att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.linear_h = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_layer = nn.Linear(self.att_hid_size, 1)
        else:
            raise NotImplementedError("To be implemented")

    def forward(self, fc_feats, att_feats, x_t_1, state):
        batch_size = fc_feats.shape[0]
        att_size = att_feats.numel() // att_feats.shape[0] // self.att_feat_size
        att = att_feats.view(-1, self.att_feat_size)
        if self.att_hid_size > 0:
            e_t_1 = self.linear_att(att).view(-1, att_size, self.att_hid_size)
            h_t_1 = self.linear_h(state[0][-1]).unsqueeze(1)
            h_t_1 = h_t_1.expand_as(e_t_1)
            e_t = F.tanh(e_t_1 + h_t_1).view(-1, self.att_hid_size)
            e_t = self.alpha_layer(e_t).view(-1, att_size)
        else:
            raise NotImplementedError("To be implemented")

        alpha_t = F.softmax(e_t, dim=0).unsqueeze(1)
        att_feats_view = att_feats.view(-1, att_size, self.att_feat_size)
        z_t = torch.bmm(alpha_t, att_feats_view).squeeze(1)

        output, state = self.rnn(torch.cat([x_t_1, z_t], dim=1).unsqueeze(0), state)
        return output.squeeze(0), state

class ShowAttendTell(nn.Module):
    def __init__(self, opt):
        super(ShowAttendTell, self).__init__()
        self.fc_feat_size = opt.fc_feat_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.dict_size = opt.dict_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_len = opt.seq_len
        self.input_encoding_size = opt.input_encoding_size

        self.rnn_core = ShowAttendTellCell(opt)
        self.linear = nn.Linear(self.fc_feat_size, self.rnn_size)
        self.word_encoding = nn.Embedding(self.dict_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.dict_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.word_encoding.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, fc_feats):
        image_map = self.linear(fc_feats).unsqueeze(0)
        if self.rnn_type == 'lstm':
            return (image_map, image_map)
        else:
            return image_map

    def forward_(self, fc_feats, att_feats, seq):
        state = self.init_hidden(fc_feats)
        outputs = []
        i_t_1 = torch.zeros_like(seq[:,0])
        for i_t in torch.unbind(seq, dim=1):
            if i_t.data.sum() == 0:
                break
            x_t_1 = self.word_encoding(i_t_1)
            self.rnn_core.rnn.flatten_parameters()
            output, state = self.rnn_core(fc_feats, att_feats, x_t_1, state)
            output = F.log_softmax(self.logit(self.dropout(output)), dim=0)
            outputs.append(output)
            i_t_1 = i_t

        for i in range(self.seq_len - len(outputs)):
            outputs.append(torch.zeros_like(outputs[0]))

        return torch.cat([o.unsqueeze(1) for o in outputs], 1)

    def forward_rnn(self, fc_feats, att_feats, i_t_1, state):
        x_t_1 = self.word_encoding(i_t_1)
        self.rnn_core.rnn.flatten_parameters()
        output, state = self.rnn_core(fc_feats, att_feats, x_t_1, state)
        output = F.log_softmax(self.logit(self.dropout(output)), dim=0)
        return output, state

    def forward(self, **kwargs):
        if not kwargs.get('seq', None) is None:
            return self.train_(**kwargs)
        else:
            return self.eval_(**kwargs)

    def train_(self, fc_feats, att_feats, seq):
        state = self.init_hidden(fc_feats)
        outputs = []
        i_t_1 = torch.zeros_like(seq[:,0])
        for i_t in torch.unbind(seq, dim=1):
            if i_t.data.sum() == 0:
                break
            output, state = self.forward_rnn(fc_feats, att_feats, i_t_1, state)
            outputs.append(output)
            i_t_1 = i_t

        for i in range(self.seq_len - len(outputs)):
            outputs.append(torch.zeros_like(outputs[0]))

        return torch.cat([o.unsqueeze(1) for o in outputs], 1)
        
    def eval_(self, fc_feats, att_feats):
        state = self.init_hidden(fc_feats)
        seqs = []
        i_t_1 = fc_feats.data.new(fc_feats.shape[0]).long().zero_()
        i_t_1 = Variable(i_t_1, requires_grad=False)
        unfinished = torch.ones_like(i_t_1).byte()
        for t in range(self.seq_len):
            output, state = self.forward_rnn(fc_feats, att_feats, i_t_1, state)

            i_t = torch.multinomial(output.exp(), 1).long().view(-1)

            unfinished = unfinished * (i_t > 0)
            if not unfinished.any():
                break

            seq = i_t * unfinished.type_as(i_t)
            seqs.append(seq)

            i_t_1 = i_t

        for i in range(self.seq_len - len(seqs)):
            seqs.append(torch.zeros_like(seqs[0]))

        return torch.cat([o.unsqueeze(1) for o in seqs], 1)
