import random
import numpy as np
import torch
from torch import optim
import math
from metric import *
import datetime
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pickle
import argparse
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='Tmal, diginetica, RetailRocket')
parser.add_argument('--epoch', type= int, default= 30 )
parser.add_argument('--ini', default='average')          # average
parser.add_argument('--name', default= 'IAN')
parser.add_argument('--GPU', type= int, default= 1 )

args = parser.parse_args()

test64 = pickle.load(open('./Dataset/{}/test.txt'.format(args.dataset), 'rb'))
train64 = pickle.load(open('./Dataset/{}/train.txt'.format(args.dataset), 'rb'))

# SEQUENCE LABEL
train64_x = train64[0]  
train64_y = train64[1] 
test64_x = test64[0]
test64_y = test64[1]


train_pos = list()
test_pos = list()
item_set = set()
# train
for items in train64[0]:
    pos = list()                
    for id_ in range(len(items)):
        item_set.add(items[id_])
        pos.append(len(items) + 1 - id_)
    pos.append(1)   # reverse target position
    train_pos.append(pos)

for item in train64[1]:
    item_set.add(item)   

# test
for items in test64[0]:
    pos = []
    for id_ in range(len(items)):
        item_set.add(items[id_])
        pos.append(len(items) + 1 - id_)
    pos.append(1)   # reverse target position
    test_pos.append(pos)

for item in test64[1]:
    item_set.add(item)


# item number begin at 1
item_list = sorted(list(item_set))
item_dict = dict()
for i in range(1, len(item_set) + 1):
    item = item_list[i - 1]
    item_dict[item] = i

# item number begin at 1
train64_x = list()
train64_y = list()
test64_x = list()
test64_y = list()

for items in train64[0]:
    new_list = []
    for item in items:
        new_item = item_dict[item]
        new_list.append(new_item)
    train64_x.append(new_list)
for item in train64[1]:
    new_item = item_dict[item]
    train64_y.append(new_item)
for items in test64[0]:
    new_list = []
    for item in items:
        new_item = item_dict[item]
        new_list.append(new_item)
    test64_x.append(new_list)
for item in test64[1]:
    new_item = item_dict[item]
    test64_y.append(new_item)

max_length = 0  # max length of session
for sample in train64_x:
    max_length = len(sample) if len(sample) > max_length else max_length
for sample in test64_x:
    max_length = len(sample) if len(sample) > max_length else max_length
print(max_length)

# build model input and output with padding 0
train_seqs = np.zeros((len(train64_x), max_length))
train_poses = np.zeros((len(train64_x), max_length+1))
test_seqs = np.zeros((len(test64_x), max_length))
test_poses = np.zeros((len(test64_x), max_length+1))

for i in range(len(train64_x)):
    seq = train64_x[i]
    pos = train_pos[i]
    length = len(seq)
    train_seqs[i][-length:] = seq
    train_poses[i][-length-1:] = pos

for i in range(len(test64_x)):
    seq = test64_x[i]
    pos = test_pos[i]
    length = len(seq)
    test_seqs[i][-length:] = seq
    test_poses[i][-length-1:] = pos

target_seqs = np.array(train64_y)
target_test_seqs = np.array(test64_y)

# raw input
train_x = torch.Tensor(train_seqs)
train_pos = torch.Tensor(train_poses)
train_y = torch.Tensor(target_seqs)
test_x = torch.Tensor(test_seqs)
test_pos = torch.Tensor(test_poses)
test_y = torch.Tensor(target_test_seqs)


class IAN(nn.Module):
    def __init__(self, item_dim, pos_dim, n_items, n_pos, w, ini, dropout=0,
                 activate='relu'):
        super(IAN, self).__init__()
        self.item_dim = item_dim
        self.pos_dim = pos_dim
        self.dim = item_dim + pos_dim
        self.n_items = n_items
        self.n_pos = n_pos + 1
        self.embedding = nn.Embedding(n_items + 1, item_dim, padding_idx=0, max_norm=1.5)
        self.pos_embedding = nn.Embedding(n_pos + 1, pos_dim, padding_idx=0, max_norm=1.5)

        self.ini = ini
        self.atten_w0 = nn.Parameter(torch.Tensor(1, self.dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.Tensor(self.dim))
        self.w_f = nn.Linear(2 * self.dim, item_dim)
        self.dropout = nn.Dropout(dropout)
        self.self_atten_w1 = nn.Linear(self.dim, self.dim)
        self.self_atten_w2 = nn.Linear(self.dim, self.dim)

        # attention refine
        self.a = nn.Linear(2 * self.item_dim, 2 * self.item_dim, bias=False)
        self.b = nn.Linear(2 * self.item_dim, 2 * self.item_dim, bias=False)
        self.a_1 = nn.Linear(2 * self.item_dim, 2 * self.item_dim, bias=False)
        self.a_2 = nn.Linear(2 * self.item_dim, 2 * self.item_dim, bias=False)
        self.a_3 = nn.Linear(4 * self.item_dim, 2 * self.item_dim, bias=False)
        self.va_t = nn.Linear(2 * self.item_dim, 1, bias=True)
        self.sf = nn.Softmax()

        self.w_f_1 = nn.Linear(self.dim, item_dim)
        self.LN = nn.LayerNorm(self.dim)
        self.attention_mlp = nn.Linear(self.dim, self.dim)
        self.alpha_w = nn.Linear(self.dim, 1)
        self.w = w
        self.is_dropout = True

        if activate == 'relu':
            self.activate = F.relu
        elif activate == 'selu':
            self.activate = F.selu

        self.initial_()

    def initial_(self):

        init.normal_(self.atten_w0, 0, 0.05)
        init.normal_(self.atten_w1, 0, 0.05)
        init.normal_(self.atten_w2, 0, 0.05)
        init.constant_(self.atten_bias, 0)
        init.constant_(self.attention_mlp.bias, 0)
        init.constant_(self.embedding.weight[0], 0)
        init.constant_(self.pos_embedding.weight[0], 0)

    def forward(self, x, pos):
        self.is_dropout = True
        x_embeddings = self.embedding(x)  # B,seq,dim
        pos_embeddings = self.pos_embedding(pos[:, :-1])  # B, seq, dim
        mask = (x != 0).float()  # B,seq

        # session-level intent representation
        x_ = torch.cat((x_embeddings, pos_embeddings), 2)  # B seq, 2*dim
        intent_s= self.self_attention(x_, x_, x_, mask)

        # target-level intent representation
        pos_predict = self.pos_embedding(pos[:, -1])
        length = torch.sum(mask, 1).unsqueeze(1).repeat((1, self.item_dim))
        average_item = torch.sum(x_embeddings, dim=1) / length
        intent_t = torch.cat((average_item, pos_predict), 1)

        # intent alignment machanism
        qs1 = self.a_1(intent_s.contiguous().view(-1, 2 * self.item_dim)).view(intent_s.size())  # key and value
        intent_t_expand = self.a_2(intent_t.unsqueeze(1).expand_as(intent_s))                    # query
        intent_t_expand_masked = mask.unsqueeze(2).expand_as(intent_s) * intent_t_expand         # query
        beta = self.va_t(torch.sigmoid(qs1 + intent_t_expand_masked).view(-1, 2 * self.item_dim)).view(mask.size())
        beta = self.sf(beta)                                                                     # alinment vector
        mutual_res = torch.sum(beta.unsqueeze(2).expand_as(intent_s) * intent_s, 1)
        
        gf = torch.sigmoid(self.a_3(torch.concat([mutual_res, intent_t], dim=-1)))
        sess = gf * intent_t + (1-gf) * mutual_res                                       # gate fusion

        result = self.decoder(sess)
        return result



    def self_attention(self, q, k, v, mask=None):
        # self attention
        if self.is_dropout:
            q_ = self.dropout(self.activate(self.attention_mlp(q)))
        else:
            q_ = self.activate(self.attention_mlp(q))
        scores = torch.matmul(q_, k.transpose(1, 2)) / math.sqrt(self.dim)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)
            scores = scores.masked_fill(mask == 0, -np.inf)
        alpha = torch.softmax(scores, dim= -1)
        # feed forward
        att_v = torch.matmul(alpha, v)  # B, seq, dim
        if self.is_dropout:
            att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(att_v)))) + att_v
        else:
            att_v = self.self_atten_w2(self.activate(self.self_atten_w1(att_v))) + att_v
        att_v = self.LN(att_v)
        return att_v


    def decoder(self, sess_l):
        if self.is_dropout:
            c = self.dropout(torch.selu(self.w_f_1(sess_l)))
        else:
            c = torch.selu(self.w_f_1(sess_l))
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        l_emb = self.embedding.weight[1:] / torch.norm(self.embedding.weight[1:], dim=-1).unsqueeze(1)
        z = self.w * torch.matmul(l_c, l_emb.t())

        return z

    def predict(self, x, pos, k=20):
        self.is_dropout = False
        x_embeddings = self.embedding(x)  # B,seq,dim
        pos_embeddings = self.pos_embedding(pos[:, :-1])  # B, seq, dim
        mask = (x != 0).float()  # B,seq

        # session-level intent representation
        x_ = torch.cat((x_embeddings, pos_embeddings), 2)  # B seq, 2*dim
        intent_s= self.self_attention(x_, x_, x_, mask)

        # target-level intent representation
        pos_predict = self.pos_embedding(pos[:, -1])
        length = torch.sum(mask, 1).unsqueeze(1).repeat((1, self.item_dim))
        average_item = torch.sum(x_embeddings, dim=1) / length
        intent_t = torch.cat((average_item, pos_predict), 1)

        # intent alignment machanism
        qs1 = self.a_1(intent_s.contiguous().view(-1, 2 * self.item_dim)).view(intent_s.size())  # key and value
        intent_t_expand = self.a_2(intent_t.unsqueeze(1).expand_as(intent_s))                    # query
        intent_t_expand_masked = mask.unsqueeze(2).expand_as(intent_s) * intent_t_expand         # query
        beta = self.va_t(torch.sigmoid(qs1 + intent_t_expand_masked).view(-1, 2 * self.item_dim)).view(mask.size())
        beta = self.sf(beta)                                                                     # alinment vector
        mutual_res = torch.sum(beta.unsqueeze(2).expand_as(intent_s) * intent_s, 1)
        
        gf = torch.sigmoid(self.a_3(torch.concat([mutual_res, intent_t], dim=-1)))
        sess = gf * intent_t + (1-gf) * mutual_res                                       # gate fusion

        result = self.decoder(sess)
        rank = torch.argsort(result, dim=1, descending=True)
        return rank[:, 0:k]

w_list = [20]
record = list()
logger = loadLogger(args.dataset, args.name)

for w in w_list:
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_sets = TensorDataset(train_x.long(), train_pos.long(), train_y.long())
    train_dataload = DataLoader(train_sets, batch_size=100, shuffle=True)
    criterion = nn.CrossEntropyLoss().cuda()
    test_x, test_pos, test_y = test_x.long(), test_pos.long(), test_y.long()
    all_test_sets = TensorDataset(test_x, test_pos, test_y)
    test_dataload = DataLoader(all_test_sets, batch_size=100, shuffle=False)

    if args.dataset == 'diginetica':
        model = IAN(100, 100, 43097, 69+1, w, args.ini, dropout=0.5).cuda()
    if args.dataset == 'RetailRocket':
        model = IAN(100, 100, 36968, 284+1, w, args.ini, dropout=0.5).cuda()
    if args.dataset == 'Tmall':
        model = IAN(100, 100, 40727, 39+1, w, args.ini, dropout=0.5).cuda()
    opti = optim.Adam(model.parameters(), lr=0.001, weight_decay=0, amsgrad=True)
    best_result = 0

    train_time = 0
    for epoch in range(args.epoch):
        logger.info('epoch: %d' % (epoch))

        start = time.time()
        start_time = datetime.datetime.now()
        # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        losses = 0
        for step, (x_train, pos_train, y_train) in enumerate(train_dataload):
            opti.zero_grad()
            q = model(x_train.cuda(), pos_train.cuda())
            loss = criterion(q, y_train.cuda()-1)
            loss.backward()
            opti.step()
            losses += loss.item()
        logger.info("mean_loss : %0.2f" % (losses))
        epoch_train_time = time.time() - start
        train_time = train_time + epoch_train_time
        print("train average Run time: %f s" % (train_time /(epoch+1)))

        end_time = datetime.datetime.now()
        with torch.no_grad():
            y_predict_list = [torch.LongTensor().cuda() for _ in range(20)]
            recall_all = [0 for _ in range(20)]
            mrr_all = [0 for _ in range(20)]
            for x_test, pos_test, y_test in test_dataload:
                with torch.no_grad():
                    y_pre = model.predict(x_test.cuda(), pos_test.cuda(), 20)
                    for i in range(20):
                        y_predict_list[i] = torch.cat((y_predict_list[i], y_pre[:, :(i+1)]), 0)
            for j in range(20):
                recall_all[j] = get_recall(y_predict_list[j], test_y.cuda().unsqueeze(1) - 1)
                mrr_all[j] = get_mrr(y_predict_list[j], test_y.cuda().unsqueeze(1) - 1)
                logger.info('\tRecall@%d:\t%.5f\tMMR@%d:\t%.5f\t' %((j+1), recall_all[j], (j+1), mrr_all[j]))

            if best_result < recall_all[-1]:
                best_result = recall_all[-1]
                torch.save(model.state_dict(), 'models/IAN[{0}-{1}].pth'.format(args.dataset, args.ini))
            logger.info("best result:%.5f " % best_result)
            logger.info("==================================")
    record.append(best_result)
    print("train Run time: %f s" % (train_time))
    print("train average Run time: %f s" % (train_time / args.epoch))
logger.info(record)
