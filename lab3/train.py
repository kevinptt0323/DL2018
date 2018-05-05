import opts
from tqdm import tqdm, trange

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import utils
from coco import CocoCaptionsFeature
from models.ShowAttendTell import ShowAttendTell
from summary import Summary

use_cuda = torch.cuda.is_available()
opt = opts.parse_opt()

data = CocoCaptionsFeature(fc_dir=opt.input_fc_dir,
                           att_dir=opt.input_att_dir,
                           label_file=opt.input_label_h5,
                           info_file=opt.input_json,
                           split=["train", "restval"],
                           opt=opt)

def expand_seq_data(data):
    _fc, _att, _labels = zip(*data)
    seq_per_img = _labels[0].shape[0]
    fc, att = [], []
    for fc_item in _fc:
        fc += [fc_item] * seq_per_img
    for att_item in _att:
        att += [att_item] * seq_per_img
    fc = torch.stack(fc)
    att = torch.stack(att)
    labels = torch.cat(_labels)
    return fc, att, labels

trainloader = DataLoader(data, batch_size=opt.batch_size,
                         num_workers=1,
                         collate_fn=expand_seq_data)

opt.dict_size = len(data.dictionary)
opt.seq_len = data.seq_len

net = ShowAttendTell(opt)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
criterion = utils.LanguageModelCriterion()

summary = Summary()

iteration = 0

def train():
    global iteration
    loader = tqdm(enumerate(trainloader), total=len(trainloader), ascii=True)
    for batch_idx, (fc, att, labels) in loader:
        if use_cuda:
            fc, att, labels = fc.cuda(), att.cuda(), labels.cuda()
        fc, att, labels = Variable(fc), Variable(att), Variable(labels)
        optimizer.zero_grad()
        outputs = net(fc, att, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        utils.clip_grad_value_(net.parameters(), opt.grad_clip)
        optimizer.step()

        train_loss = loss.data[0]

        loader.set_description("Loss: {:.6f}".format(train_loss))

        iteration += 1
        if iteration % opt.losses_log_every == 0:
            summary.add(iteration, 'train-loss', train_loss)

for epoch in trange(opt.max_epochs, desc='Epoch', ascii=True):
    train()

summary.write("csv/history.csv")
