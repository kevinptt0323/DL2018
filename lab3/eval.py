import opts
import os
from tqdm import tqdm, trange

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import utils
from coco import CocoCaptionsFeature
from models.ShowAttendTell import ShowAttendTell

use_cuda = torch.cuda.is_available()
opt = opts.parse_opt()

data = CocoCaptionsFeature(fc_dir=opt.input_fc_dir,
                           att_dir=opt.input_att_dir,
                           label_file=opt.input_label_h5,
                           info_file=opt.input_json,
                           split="val",
                           opt=opt)

dataloader = DataLoader(data, batch_size=opt.batch_size, num_workers=1)

opt.dict_size = len(data.dictionary)
opt.seq_len = data.seq_len

net = ShowAttendTell(opt)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

if vars(opt).get('start_from', None) is not None:
    state_dict = torch.load(os.path.join(opt.start_from, 'model.pth'))
    if not use_cuda:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = new_state_dict
    net.load_state_dict(state_dict)

criterion = utils.LanguageModelCriterion()

def test():
    net.eval()

    loader = tqdm(enumerate(dataloader), total=len(dataloader), ascii=True)

    min_loss = 1e9

    for batch_idx, (fc, att, labels, data_info) in loader:
        if use_cuda:
            fc, att, labels = fc.cuda(), att.cuda(), labels.cuda()
        fc, att, labels = Variable(fc, requires_grad=False), Variable(att, requires_grad=False), Variable(labels, requires_grad=False)
        fc = torch.stack([fc]*opt.seq_per_img).view(-1, *fc.shape[1:])
        att = torch.stack([att]*opt.seq_per_img).view(-1, *att.shape[1:])
        origin_labels = labels.view(-1, *labels.shape[2:])
        labels = labels.transpose(1, 0).contiguous().view(-1, *labels.shape[2:])

        labels = labels.long()
        outputs, _ = net(fc_feats=fc, att_feats=att, seq=labels)
        loss = criterion(outputs, labels)

        if loss.data[0] < min_loss:
            min_loss = loss.data[0]

            outputs, alpha = net(fc_feats=fc, att_feats=att)
            min_txts = utils.decode_sequence(data.dictionary, outputs.data)
            min_txts_target = utils.decode_sequence(data.dictionary, origin_labels.data)
            file_path = data_info['file_path']

        loader.set_description("Loss: {:.6f} | Min Loss: {:.6f}".format(loss.data[0], min_loss))

        if min_loss < 1.54:
            break

    loader.set_description("Loss: {:.6f} | Min Loss: {:.6f}".format(loss.data[0], min_loss))

    for idx, (txt) in enumerate(min_txts):
        if idx % opt.seq_per_img == 0:
            print(file_path[idx // opt.seq_per_img])
        print(txt)
        print(min_txts_target[idx])
        if idx % opt.seq_per_img == 4:
            print("")

    print(min_loss)
    att_path = './alpha.pt'
    torch.save(alpha.data.cpu(), att_path)

test()

