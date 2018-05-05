import torch
import torch.nn as nn
import torch.nn.functional as F

class ShowAttendTellCell(nn.Module):
    def __init__(self, opt):
        super(ShowAttendTellCell, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.rnn = getattr(nn, self.rnn_type.upper())(self.att_feat_size, 
                self.rnn_size, bias=False, dropout=self.drop_prob_lm)

        if self.att_hid_size > 0:
            self.linear_att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.linear_h = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_layer = nn.Linear(self.att_hid_size, 1)
        else:
            raise NotImplementedError("To be implemented")

    def forward(self, fc_feats, att_feats, word, state):
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

        output, state = self.rnn(z_t.unsqueeze(0), state)
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

        self.rnn_core = ShowAttendTellCell(opt)
        self.linear = nn.Linear(self.fc_feat_size, self.rnn_size)
        #self.embed = nn.Embedding(self.dict_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.dict_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

    def init_hidden(self, fc_feats):
        image_map = self.linear(fc_feats).unsqueeze(0)
        if self.rnn_type == 'lstm':
            return (image_map, image_map)
        else:
            return image_map

    def forward(self, fc_feats, att_feats, seq):
        self.rnn_core.rnn.flatten_parameters()
        state = self.init_hidden(fc_feats)
        outputs = []
        for word in seq.split(1, dim=1):
            if word.data.sum() == 0:
                break
            output, state = self.rnn_core(fc_feats, att_feats, word, state)
            output = F.log_softmax(self.logit(self.dropout(output)), dim=0)
            outputs.append(output)

        for i in range(self.seq_len - len(outputs)):
            outputs.append(torch.zeros_like(outputs[0]))

        return torch.cat([o.unsqueeze(1) for o in outputs], 1)

