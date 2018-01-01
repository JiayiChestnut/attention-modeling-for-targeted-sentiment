
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 02:55:28 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 22:52:08 2017

@author: Administrator
"""

import argparse
import time
import numpy as np 
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import utils as utils
import shutil
#
#parser = argparse.ArgumentParser(description='PyTorch LSTM Training')
#parser.add_argument('-d', '--drop', metavar='P', default = 0, type = float,
#                    help='dropout probability')
#parser.add_argument('-b', '--batchsize', metavar='N', default= 10, type = int)
#parser.add_argument('--lr', '-l', metavar = 'N', default = 0.1, type = float)
#args = parser.parse_args()
#batch_size = args.batchsize
#pdrop = args.drop
#lr = args.lr


lr = 0.1 #learning rate
pdrop = 0 #probability of dropout
batch_size = 10 #10 sentences for one time bp

#gpu_avail = torch.cuda.is_available()
gpu_avail = False
calculate = True
print_freq = 10
is_debug = False
num_epoch = 90
is_evaluate = False


embed_dim, hidden_dim, att_hidden_dim, output_dim = (200, 150, 100, 3)
#1. data processing
def prepare_data(filename):
    global batch_size
    with open(filename, 'r', encoding = 'utf8') as f:
        lines = f.readlines()
    word_to_ix_file = "attn_word_to_ix.npy"
    ix_to_vector_file = "attn_ix_to_vector.npy"
    word_to_ix = np.load(word_to_ix_file).item()
    ix_to_vector = np.load(ix_to_vector_file).item()
    tag_to_ix = {"-1":0, "0":1, "1":2}
#    
#    sens = []
#    left_idx = []
#    right_idx = []
#    labels = []
    train_data = []
    sub_sens = []  #one sub sentence contains 10 sentences
    sub_labels = []
    sub_lidx = []
    sub_ridx = []
    for i, line in enumerate(lines):
        
        sen, idxs, label = line.split("|||")
        idxs = idxs.strip().split(" ")  #delete space
        sub_sens.append(prepare_sentence(sen, 
                                      word_to_ix, 
                                      ix_to_vector))
        sub_lidx.append(int(idxs[0]))
        sub_ridx.append(int(idxs[1]))
        sub_labels.append(prepare_tag(tag_to_ix[label.strip()]))
        if (i % batch_size == 0):
#            sens.append(sub_sens)
#            left_idx.append(sub_lidx)
#            right_idx.append(sub_ridx)
#            labels.append(sub_labels)
#           
            train_data.append(list(zip(sub_sens, sub_labels, sub_lidx, sub_ridx)))
            sub_sens = []
            sub_labels = []
            sub_lidx = []
            sub_ridx = []
            
#        print(sen)
#        print(start_idx, end_idx)
#        print(label)
    
    
       #here tensordatas consist of inputsize row, each row a 3 element list, (tensorlidx, tensorridx, tensorsen)
#    data = list(zip(list_tensorsens, list_tensortags,left_idx, right_idx))
    return train_data

#1. data processing
#def prepare_data(filename):
#    with open(filename, 'r', encoding = 'utf8') as f:
#        lines = f.readlines()
#    word_to_ix_file = "attn_word_to_ix.npy"
#    ix_to_vector_file = "attn_ix_to_vector.npy"
#    word_to_ix = np.load(word_to_ix_file).item()
#    ix_to_vector = np.load(ix_to_vector_file).item()
#    tag_to_ix = {"-1":0, "0":1, "1":2}
#    sen_tensor = []
#    labels = []
#    for line in lines:
#        sen, idxs, label = line.split("|||")
#        idxs = idxs.strip().split(" ")
#        sen_tensor.append(prepare_sentence(sen, 
#                                      word_to_ix, 
#                                      ix_to_vector))
#                                     
#        labels.append([tag_to_ix[label.strip()], 
#                                 int(idxs[0]), 
#                                int(idxs[1])])
##        print(sen)
##        print(start_idx, end_idx)
##        print(label)
#    
#    
#    tensordatas = torch.stack(sen_tensor)# if we do it this way ,then cause inconsisten tensor size
#    tensortags = torch.stack()
#    #here tensordatas consist of inputsize row, each row a 3 element list, (tensorlidx, tensorridx, tensorsen)
##    data = list(zip(list_tensorsens, list_tensortags,left_idx, right_idx))
#    my_dataset = torch.utils.data.TensorDataset(tensordatas, tensortags)
#    my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size = bsize)
#    return my_dataloader


def prepare_sentence(seq, word_to_ix, ix_to_vector):
    idxs = [ix_to_vector[word_to_ix[w.lower()]] for w in seq.split(" ")]
    tensor = torch.from_numpy(np.asarray(idxs)).float()# change tensor element type into float
    if gpu_avail:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def prepare_sen_idx(seq, word_to_ix, ix_to_vector, startidx, endidx):
    # can't work since each element in word_var is a np.ndarray, but startidx and endidx just int
    idxs = [ix_to_vector[word_to_ix[w.lower()]] for w in seq.split(" ")]
    print(idxs[0])
    print(idxs[0])
    print(startidx)
    print(endidx)
    idxs = np.concatenate((np.asarray(idxs),
                           np.array([int(startidx), int(endidx)])))
                           
    
    tensor = torch.from_numpy(np.asarray(idxs))
    if gpu_avail:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def prepare_tag_idx(tag_idx):
    tensor = torch.from_numpy(np.asarray(tag_idx)).long()
    if gpu_avail:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)
    
def prepare_tag(tag):
    tensor = torch.from_numpy(np.asarray([tag])).long()
    if gpu_avail:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)
def prepare_idx(idx):
    tensor = torch.from_numpy(np.asarray(idx))
    if gpu_avail:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)
    #here sens is a string of sentence, idx include start index and stop index
    #of the target, label is a number

#2. BiLSTM to produce output(learn how the encoder and decoder train)

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, att_hidden_dim, output_dim):
        #https://github.com/vanzytay/pytorch_sentiment_rnn/blob/master/models/td_rnn.py
        
        super(AttentionModel, self).__init__()
        # so here we have only one hidden state
        self.hidden_dim = hidden_dim
        self.p_dropout = pdrop# in original c code, dropout only use in word embedding variable
        self.dropout = nn.Dropout(self.p_dropout)
        self.l2rlstm = nn.LSTM(embed_dim, hidden_dim)
        self.r2llstm = nn.LSTM(embed_dim, hidden_dim)
        
        self.input2weight1 = nn.Linear(4*hidden_dim, att_hidden_dim)
        
        self.input2weight2 = nn.Linear(att_hidden_dim, 1)# output a weight for each words
        
        self.weight_softmax = nn.Softmax()
        
        self.attn_out = nn.Tanh()
        
        self.applied2out = nn.Linear(2*hidden_dim, output_dim)
        self.out = nn.Softmax()
        
        self.init_weights()
        
        self.hidden = self.init_hidden()
        
        self.clip_grad = 5
        self.is_test = False#when testing should close up
        
        print("successfully initialize model--")
    def init_weights(self):
        initrange = 0.1
        self.input2weight1.weight.data.uniform_(-initrange, initrange)
        self.input2weight1.bias.data.fill_(0)
        self.input2weight2.weight.data.uniform_(-initrange, initrange)
        self.input2weight2.bias.data.fill_(0)
        self.applied2out.weight.data.uniform_(-initrange, initrange)
        self.applied2out.bias.data.fill_(0)
    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))
        # each time we input a whole sentence to LSTM model, we need to 
        #initialize the hidden state for it.
        # since lstm need inputs like : input, (h_0, c_0)
        
    def forward(self, words_var, startidx, endidx):
        global is_debugg
        global gpu_avail
    #words is a list of word embedding
#        rwords = reversed(words)
#        words_tens= torch.from_numpy(np.asarray(words))
#        rwords_tens = torch.from_numpy(np.asarray(rwords))
#        if gpu_avail:
#            words_tens.cuda()
#            rwords_tens.cuda()
#        words_var = Variable(words_tens)
#        rwords_var = Variable(rwords_tens)
        
    # make words_var is a list of var
       # input words_var come from wordnum x wordembed_dim
        #get a reverse words var
        
#        words_var = words_var.unsqueeze(0)
#        words_var = self.dropout(words_var)#??? drop out 应该放在
        num_words = words_var.size()[0]
        num_targets = endidx - startidx + 1# 10, 12 means only take 10th, 11th element
        num_non_targets = num_words - num_targets
        idx = list(range( num_words- 1, -1, -1))
        idx = Variable(torch.LongTensor(idx))
        if gpu_avail:
            idx = idx.cuda()
#            print("here cudaing")
        rwords_var = words_var.index_select(0, idx)
        # 12 12 mean 0-based 12th element is the target word
        
        leftorder = list(range(startidx)) + list(range(endidx + 1, num_words))
        #assume that leftoder is [0,1, 4,5,6]
        leftidx_var = Variable(torch.LongTensor(leftorder))#be careful that LongTensor should take 
        rightidx_var = Variable(torch.LongTensor([num_words - n - 1 for n in leftorder]))
        #rightidx = [6,5 , 2, 1, 0]
        targetorder = list(range(startidx, endidx + 1))
        l_targetidx_ten = Variable(torch.LongTensor(targetorder))
        r_targetidx_ten = Variable(torch.LongTensor([num_words - n - 1 for n in targetorder]))
        if gpu_avail:
            leftidx_var.cuda()
            rightidx_var.cuda()
            l_targetidx_ten.cuda()
            r_targetidx_ten.cuda()
        
    
        l2routputs, _ = self.l2rlstm(words_var, self.hidden)#since lstm only have 1 layer,output are all hiddenlayer
        r2loutputs, _ = self.r2llstm(rwords_var, self.hidden)
        
        l2routputs = l2routputs.squeeze(1)#become x1 x 150(hd)
        r2loutputs = r2loutputs.squeeze(1)
        if is_debug:
            print("l2routput size:{}, r2loutput:{}".format(l2routputs.size,r2loutputs.size()))
            #torch.Size([15, 1, 150]) torch.Size([15, 1, 150])
            print("start index:{}, end index:{}, num_words:{}".format(startidx, endidx, num_words))
        
        targetoutputs = torch.cat((l2routputs.index_select(0, l_targetidx_ten),
                                  r2loutputs.index_select(0, r_targetidx_ten)), 1)
        if is_debug:
            print("before mean, targetoutput size:", targetoutputs.size())# this is a 3 dim tensor

#        targetoutputs = targetoutputs.squeeze(0)
#        print(targetoutputs.size())# this is a 3 dim tensor
        targetoutputs = targetoutputs.mean(0).view(1, -1)# finally it is a 1xhidden_dim tensor
        if is_debug:
            print("after mean: targetoutput size:", targetoutputs.size())# this is a 3 dim tensor
            print("num_no_targets:", num_non_targets)
        targetoutputs = targetoutputs.repeat(num_non_targets,1)# in order to append it into each input
        if is_debug:
            print("after repeat, targetoutputs:", targetoutputs.size())# this is a 3 dim te
        # need to get the target part then normal part
        l2rinputs = l2routputs.index_select(0, leftidx_var)
        r2linputs = r2loutputs.index_select(0, rightidx_var)

        
        if is_debug:
            print("l2routput size:{}, r2loutput:{}".format(l2routputs.size(),r2loutputs.size()))
        inputs = torch.cat((l2rinputs, r2linputs), 1)
        if is_debug:
            print("inputs: ", inputs.size())
        inputs_targets = torch.cat((inputs, targetoutputs), 1)
        if is_debug:
            print("input_target", inputs_targets.size())
        # this inputs is a size of num_non_targets x (150+150+150)
        
        attn_hd = self.input2weight1(inputs_targets)
        if is_debug:
            print("attn_hd:", attn_hd.size())
        attn_hd = self.attn_out(attn_hd)
#        if self.is_test:
#            attn_hd = self.dropout(attn_hd)
        if is_debug:
            print("attn_hd:", attn_hd.size())
        attn_w = self.input2weight2(attn_hd)
#        if self.is_test:
#            attn_w = self.dropout(attn_w)
        if is_debug:
            print("attn_w:", attn_w.size())
        attn_norm_w = self.weight_softmax(attn_w)
        if is_debug:
            print("normalized_weight:", attn_norm_w.size())
        
        attn_weights_applied = torch.bmm(attn_norm_w.t().unsqueeze(0), 
                                         inputs.unsqueeze(0)).squeeze(0)
        if self.is_test:
            attn_weights_applied = self.dropout(attn_weights_applied)
        if is_debug:
            print("attn_weights_applied", attn_weights_applied.size())
        output = self.applied2out(attn_weights_applied)
        if self.is_test:
            output = self.dropout(output)
        if is_debug:
            print("output", output.size())
        output = self.out(output)
        return output
        
        
        
#3. attention layer

#4. design the train process
def main():
    global is_evaluate #skip training data
    global lr
    best_prec1 = 0
    
    global num_epoch 
    
    global embed_dim, hidden_dim, att_hidden_dim, output_dim
    
    
    model = AttentionModel(embed_dim, hidden_dim, att_hidden_dim, output_dim)
    if gpu_avail:
        model.cuda()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if gpu_avail:
        criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr)
    
    train_file = "ttrain.posneg"
#    train_file = "ttest.posneg"
    test_file = "ttest.posneg"
    train_data = prepare_data(train_file)
    test_data = prepare_data(test_file)
    
    
    
# =============================================================================
    if is_evaluate:
        validate(test_data, model, criterion)
        return
# =============================================================================

    for epoch in range(num_epoch):
        
#        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_data, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(test_data, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(training_data, model, criterion, optimizer, epoch):
    global calculateg
    global print_freq
    global batch_size
    #training data is a list, whose element is (sens, label)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    model.is_test = False

    end = time.time()
    len_data = sum([len(i) for i in training_data])
    for batch_id, batch_data in enumerate(training_data):
        
        for i, (sentences_var, tags_var, startidx, endidx) in enumerate(batch_data):
    #        tags_var = datapart.index_select(0, torch.LongTensor([0]))
    #        startidx = datapart.index_select(0, torch.LongTensor([1]))
    #        endidx = datapart.index_select(0, torch.LongTensor([2]))
    #        data_time.update(time.time() - end)
    #        # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()#??? zero grad 放在batchsize 外
    
            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()
            
            
        
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.
            
            # Step 3. Run our forward pass.
            output = model(sentences_var, startidx, endidx)
            prec1, prec2 = accuracy(output, tags_var, topk = (1,2))
#            print("prec1", float(prec1.data.numpy()))
            top1.update(float(prec1.data.numpy()), batch_size)
#            top2.update(float(prec2.data.numpy()), batch_size)
            
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = criterion(output, tags_var)
    #        if calculate:
    #            prec1, prec5 = accuracy(output.data, tags_var, topk=(1, 3))
            if i == 0:
                temp_loss_sum = loss
            else:
                temp_loss_sum += loss
            losses.update(loss.data[0], sentences_var.size(0))
#        if calculate:
#            top1.update(prec1[0], sentences_var.size(0))
#            top5.update(prec5[0], sentences_var.size(0))
        
        temp_loss_sum.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), model.clip_grad)
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
#        if i % print_freq == 0:
#            print("train---epoch:{}, loss:{}".format(epoch, loss.data[0]))
        
        if calculate & (batch_id % print_freq == 0):
            print('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top_1.val:.3f} ({top_1.avg:.3f})\t'
                  'Prec@5 {top_5.val:.3f} ({top_5.avg:.3f})'.format(
                   batch_id*batch_size, len_data, batch_time=batch_time, loss=losses,
                   top_1=top1, top_5=top5))



        
#    
#    
#
def validate(test_data, model, criterion):
    global batch_size
    global print_freq
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.is_test = True

    end = time.time()
    len_data = sum([len(i) for i in test_data])
    for batch_id, batch_data in enumerate(test_data):
                
        for i, (sentences_var, tags_var, startidx,endidx) in enumerate(batch_data):
            data_time.update(time.time() - end)
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
    
            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()
        
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.
            
        
            # Step 3. Run our forward pass.
            output = model(sentences_var, startidx, endidx)
            prec1, prec2 = accuracy(output, tags_var, topk = (1,2))
            
            top1.update(float(prec1.data.numpy()), batch_size)
            top5.update(float(prec2.data.numpy()), batch_size)
            
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = criterion(output, tags_var)
    #        if calculate:
    #            prec1, prec5 = accuracy(output.data, tags_var, topk=(1, 3))
    #        print("test----epoch:{}, loss:{}".format(epoch, loss.data[0]))
            losses.update(loss.data[0], sentences_var.size(0))
    #        if calculate:
    #            top1.update(prec1[0], sentences_var.size(0))
    #            top5.update(prec5[0], sentences_var.size(0))
    
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
        if calculate & (batch_id % print_freq == 0):
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top_1.val:.3f} ({top_1.avg:.3f})\t'
                      'Prec@5 {top_5.val:.3f} ({top_5.avg:.3f})'.format(
                       batch_id*batch_size, len_data, batch_time=batch_time, loss=losses, top_1=top1, top_5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
#    print("output\n", output)
#    print("target\n", target)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == "__main__":
    main()