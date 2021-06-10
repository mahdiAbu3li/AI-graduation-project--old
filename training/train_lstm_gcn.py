# --coding:utf-8--

import torch
from torch import optim
from torch.nn import functional as F
import os
import time
import random
import numpy as np
from training.data import preprocess_adj, load_single_graph4lstm_gcn
from training.model import LSTM_GCN
from training.model import GCN
# from config import args
from training.utils import masked_acc, weighted_loss, cal_accuracy

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)
device = torch.device('cuda:0')


def convert_sparse_train_input1(adj, features):
    supports = preprocess_adj(adj)
    # print(features, "sss")
    m = torch.from_numpy(supports[0]).long()
    n = torch.from_numpy(supports[1])
    support = torch.sparse.FloatTensor(m.t(), n, supports[2]).float()
    # print(support)
    # features = [torch.tensor(idxs, dtype=torch.long).to(device) if torch.cuda.is_available() else \
    #                 torch.tensor(idxs, dtype=torch.long) for idxs in features]

    # i = torch.from_numpy(features[0]).long()
    # v = torch.from_numpy(features[1])
    # feature = torch.sparse.FloatTensor(i.t(), v, features[2])
    features = torch.FloatTensor(features)
    if torch.cuda.is_available():
        m = m.to(device)
        n = n.to(device)
        # support = torch.sparse.FloatTensor(m.t(), n, supports[2]).float().to(device)
        support = torch.sparse.FloatTensor(m.t(), n, supports[2]).float().to(device)
        # i = i.to(device)
        # v = v.to(device)
        # feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to(device)
    # print("**", features[0].dtype)
    # print("**", support.dtype)

    return features, support


def convert_loss_input(y_train, weight_mask):
    train_label = torch.from_numpy(y_train).long()
    weight_mask = torch.from_numpy(weight_mask)

    if torch.cuda.is_available():
        train_label = train_label.to(device)
        weight_mask = weight_mask.to(device)

    train_label = train_label.argmax(dim=1)

    return train_label, weight_mask


# print('x :', feature)
# print('sp:', support)
# num_features_nonzero = feature._nnz()
# feat_dim = feature.shape[1]

# # net = GCN(feat_dim, num_classes, num_features_nonzero)
net = GCN(22, 4)
# net = LSTM_GCN(embedding_dim=64, hidden_dim=64, vocab_size=1076, output_dim=5)
# if torch.cuda.is_available():
#     net = net.to(device)
# # optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
optimizer = optim.Adam(net.parameters(), lr=0.01)
net.train()

# 读取文件
img_dir = "../data/train_images"
test_dir = "../data/test_images"
file_list = []
test_list = []
for file in os.listdir(img_dir):
    file_list.append("../data/matrix_data_train/" + file[:-4])

for file in os.listdir(test_dir):
    test_list.append("../data/matrix_data_test/" + file[:-4])

val_list = random.sample(file_list, 1)
train_list = list(set(file_list) - set(val_list))
print("train_data : ", len(train_list), train_list)
print("test_data : ", len(test_list), train_list)

# exit()
#epoch
for epoch in range(100):
    random.shuffle(train_list)
    t1 = time.time()
    for file_name in train_list:
        adj, features, train_labels, weight_mask = load_single_graph4lstm_gcn(file_name)
        feature, support = convert_sparse_train_input1(adj, features)
        train_labels, weight_mask = convert_loss_input(train_labels, weight_mask)
        out = net((feature, support))
        out = out[0]
        # loss = masked_loss(out, train_labels, weight_mask)
        loss = weighted_loss(out, train_labels, weight_mask)
        # print("cross entropy loss: {:.5f} ".format(loss.item()))
        # loss += args.weight_decay * net.l2_loss()
        loss += 10 * net.l2_loss()

        # acc = masked_acc(out, train_labels)
        acc = cal_accuracy(out, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        if (epoch + 1) % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "time: {:.5f}, loss: {:.5f}, acc: {:.5f}". \
                  format((t2 - t1), loss.item(), acc.item()))

    # if epoch % 10 == 0:
    #     print(epoch, loss.item(), acc.item())

net.eval()

result_list = []
acc_list = []
loss_list = []
predict_label = []
real_label = []

# for file_name in val_list:
# for file_name in train_list:
for file_name in test_list:
    result_list.append("=========" + file_name + "===========")
    # adj, features, test_labels, weight_mask = load_single_graph(file_name)
    # feature, support = convert_sparse_train_input1(adj, features)
    adj, features, test_labels, weight_mask = load_single_graph4lstm_gcn(file_name)
    print(test_labels)
    feature, support = convert_sparse_train_input1(adj, features)
    # train_labels, weight_mask = convert_loss_input(test_labels, weight_mask)
    out = net((feature, support))
    out = out[0]
    pred = out.argmax(dim=1)
    # print(pred)
    label_cs = ['name', 'phone', 'total', 'o']
    p_label = [label_cs[i.item()] for i in pred]
    r_label = [label_cs[np.argmax(i)] for i in test_labels]

    predict_label.extend([i.item() for i in pred])
    real_label.extend([np.argmax(i) for i in test_labels])
    # print("*******", file_name,"*******")
    # for i,j in zip(p_label, r_label):
    #     print (i, "----",j)
    #     result_list.append(i)
    #     result_list.append(j)

    print(predict_label)
from sklearn.metrics import classification_report


# print(classification_report(real_label, predict_label, target_names=label_cs))
