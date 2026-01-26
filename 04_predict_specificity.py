import torch
import sys
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import re
token = re.compile('[A-Za-z]')
# if hasattr(torch.cuda,'empty_cache'):
#     torch.cuda.empty_cache()
torch.cuda.manual_seed(1234)
import random
seed = 1234
random.seed(seed)

from torchtext.vocab import build_vocab_from_iterator
print('------------------------0提取数据-------------------------------')
import re
token = re.compile('[A-Za-z]')
def reg_text(sequence):
    new_text = token.findall(sequence)
    new_text = [word for word in new_text]
    return new_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device',device)
data = pd.read_csv(r'1/pythonProject4/3-win/2023/model/pycham2023/py_amp_stand_sqe.csv', index_col=False)
print(len(data))
data.columns=data.columns.str.lower()
list2=[]
for i in data['sequence']:
    i=''.join(i)
    list2.append(i)
data['sequence']=list2
data.sort_values(by=['sequence'],ascending=True,inplace=True)

data=data.drop_duplicates(subset=["sequence"],keep='first',ignore_index=True)
data =data[data.notnull()]

print(len(data))
data=data[data['length']>4]
data=data[data['length']<51]
data = data[['sequence','target']]
data['sequence'] = data.sequence.apply(reg_text)
data['target']= data['target'].apply(lambda x :
                                     1 if x=='AMP'
                                     else 0)
# data['target'] = pd.factorize(data.target)[0]
print(data.head(2))
print(torch.cuda.is_available())
#创建分词列表
def yield_tokens(data):
    for text in data:
        yield text

from torchtext.vocab import build_vocab_from_iterator
vocab = build_vocab_from_iterator(yield_tokens(data.sequence),specials=['<pad>','<unk>'])
vocab.set_default_index(vocab["<unk>"])
vocab_size = len(vocab)
print(vocab_size)
print(vocab['F'])


print('------------------------LSTM模型-------------------------------')
embeding_dim = 300
hidden_size = 70
vocab_size = len(vocab)

class LSTM_Net(nn.Module):
    def __init__(self, vocab_size, embeding_dim):
        super(LSTM_Net, self).__init__()
        self.em = nn.Embedding(vocab_size, embeding_dim)  # batch*maxlen*embed_dim
        self.lstm = nn.LSTM(embeding_dim, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*70, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.em(x)
        x = x.view(len(x), -1, embeding_dim)
        x, _ = self.lstm(x)  # x——》 batch, time_step, output
        # print(x.shape)
        x = x.contiguous().view(len(x), -1)
        x = F.dropout(F.relu(self.fc1(x)),p=0.8)
        x = F.dropout(F.relu(self.fc2(x)),p=0.4)
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_Net(vocab_size, embeding_dim)
PATH_model = 'bin/HWLSTM.pth'
model.load_state_dict(torch.load(PATH_model))
model.to(device)
# print(model)


print('------------------------2载入预测数据-------------------------------')
PATH_='bin/test-hw-lstm.csv'
# PATH_='1/pythonProject4/3-win/2023/model/pycham2023/filter_analyse/TSPs/compare/Scp_ToxinPred.xlsx'
da_= pd.read_csv(PATH_,index_col=False,)
# da_.to_excel(r"1/pythonProject4/3-win/2023/model/pycham2023/DL/mic0123/hw-dl/pic/test-hw-lstm.xlsx",
#              index=False)
# PATH_=r"1/pythonProject4/3-win/2023/model/pycham2023/DL/mic0123/hw-dl/pic/test-hw-lstm.xlsx"
# da_= pd.read_excel(PATH_,index_col=False,)

# da_ = da_[(da_['length']<51)&(da_['length']>4)]
# # da_=da_[:1000]
# da_['sequence']
da=pd.DataFrame()
da['sequence']=da_['sequence']#构建数据
da['sequence']=da.sequence.apply(reg_text)

x_list= []
for x in da.sequence:
    x = vocab(x)
    x = x + [0] * (70-len(x))
    x= torch.tensor(x, dtype=torch.int64)
    x_list.append(x)
x_list=torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True)
x_list=x_list.to(device)
print(x_list.shape)
print(x_list)
print('------------------------模型预测-------------------------------')
# model.forward(x_list)
model.eval().to(device)
y_pred_= model(x_list)
y_pred_= torch.argmax(y_pred_,dim=1)
y_pred = y_pred_
print((torch.unique(y_pred,return_counts=True)))
print(y_pred.shape)
y_pred = y_pred.cpu().numpy().tolist()
print('------------------------保存数据-------------------------------')
data=pd.read_csv(PATH_,index_col=False)
# data = data[(data['length']<51)&(data['length']>4)]
data['AWLSTM']=y_pred
print(data.columns)
print(data.head(2))
# data.to_csv(PATH_,index=False)
print('------------------------打印词表------------------------------')
# print("词表",vocab(['H', 'D','R', 'P','A', 'C','G', 'Q','E', 'K','L', 'M','N', 'S','Y', 'T','I', 'W','P', 'V',]))
# #acc
# print('------------------------Accuracy------------------------------')
# from  sklearn.metrics import  accuracy_score
# if 1 in data['target']:
#     data['target']= data['target'].apply(lambda x :
#                                          1 if x=='AMP'
#                                          else 0)
# y_test = data['target'].values
# acc= accuracy_score(y_test,y_pred)
# print('acc=',round(acc,2))
# data['AWLSTM_Accuracy'] = acc
# data.to_excel(PATH_,index=False)
# print(data.shape)
# print(data[:2])
