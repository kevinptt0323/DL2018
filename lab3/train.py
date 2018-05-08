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
from summary import Summary

use_cuda = torch.cuda.is_available()
opt = opts.parse_opt()

data = CocoCaptionsFeature(fc_dir=opt.input_fc_dir,
                           att_dir=opt.input_att_dir,
                           label_file=opt.input_label_h5,
                           info_file=opt.input_json,
                           split=["train", "restval"],
                           opt=opt)

trainloader = DataLoader(data, batch_size=opt.batch_size,
                         num_workers=1)

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
train_loss = 0
train_loss_iter = 0
min_loss = 1e9

def train():
    global iteration, train_loss, train_loss_iter, min_loss
    net.train()
    loader = tqdm(enumerate(trainloader), total=len(trainloader), ascii=True)
    for batch_idx, (fc, att, labels, data_info) in loader:
        if use_cuda:
            fc, att, labels = fc.cuda(), att.cuda(), labels.cuda()
        fc, att, labels = Variable(fc, requires_grad=False), Variable(att, requires_grad=False), Variable(labels, requires_grad=False)
        fc = torch.stack([fc]*opt.seq_per_img).view(-1, *fc.shape[1:])
        att = torch.stack([att]*opt.seq_per_img).view(-1, *att.shape[1:])
        labels = labels.transpose(1, 0).contiguous().view(-1, *labels.shape[2:])

        labels = labels.long()
        optimizer.zero_grad()
        outputs, *_ = net(fc_feats=fc, att_feats=att, seq=labels)
        loss = criterion(outputs, labels)
        loss.backward()
        utils.clip_grad_value_(net.parameters(), opt.grad_clip)
        optimizer.step()

        train_loss += loss.data[0]
        train_loss_iter += 1
        min_loss = min(min_loss, loss.data[0])

        loader.set_description("Loss: {:.6f} | Min Loss: {:.6f}".format(loss.data[0], min_loss))

        iteration += 1
        if iteration % opt.losses_log_every == 0:
            summary.add(iteration, 'train-loss', train_loss / train_loss_iter)
            summary.add(iteration, 'min-loss', min_loss)
            train_loss = 0
            train_loss_iter = 0

    checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
    optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
    if use_cuda:
        torch.save(net.module.state_dict(), checkpoint_path)
    else:
        torch.save(net.state_dict(), checkpoint_path)
    torch.save(optimizer.state_dict(), optimizer_path)


def val(split="val"):
    net.eval()
    data_val = CocoCaptionsFeature(fc_dir=opt.input_fc_dir,
                               att_dir=opt.input_att_dir,
                               label_file=opt.input_label_h5,
                               info_file=opt.input_json,
                               split=split,
                               opt=opt)
    evalloader = iter(DataLoader(data_val, batch_size=opt.val_images_use, num_workers=1))

    #loader = tqdm(enumerate(trainloader), total=len(trainloader), ascii=True)
    fc, att, labels = next(evalloader)

    if use_cuda:
        fc, att, labels = fc.cuda(), att.cuda(), labels.cuda()
    fc, att, labels = Variable(fc, requires_grad=False), Variable(att, requires_grad=False), Variable(labels, requires_grad=False)
    fc = torch.stack([fc]*opt.seq_per_img).view(-1, *fc.shape[1:])
    att = torch.stack([att]*opt.seq_per_img).view(-1, *att.shape[1:])
    labels = labels.transpose(1, 0).contiguous().view(-1, *labels.shape[2:])

    labels = labels.long()
    outputs, *_ = net(fc_feats=fc, att_feats=att)
    #loss = criterion(outputs, labels)

    txts = utils.decode_sequence(data.dictionary, outputs.data)
    for txt in txts:
        print(txt)

    #train_loss += loss.data[0]
    #train_loss_iter += 1
    #min_loss = min(min_loss, loss.data[0])


for epoch in trange(opt.max_epochs, desc='Epoch', ascii=True):
    train()

# val()

summary.write("csv/history5-6.csv")
