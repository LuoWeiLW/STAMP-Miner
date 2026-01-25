import torch
from torch.utils.data import DataLoader

torch.cuda.manual_seed(1234)
import random
seed = 1234
random.seed(seed)
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

from torchtext.vocab import build_vocab_from_iterator
print('------------------------0提取数据-------------------------------')
import re
token = re.compile('[A-Za-z]')
def reg_text(sequence):
    new_text = token.findall(sequence)
    new_text = [word for word in new_text]
    return new_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv(r'D:\fzu\lw\jupyter\pycham2023\amps_predictor\py_amp_stand_sqe.csv', index_col=False)
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
print('------------------------1划分训练与测试集创建train_data和test_data------------------------------')
print("词表",vocab(['H', 'D','R', 'P','A', 'C','G', 'Q','E', 'K','L', 'M','N', 'S','Y', 'T','I', 'W','P', 'V',]))
list1=vocab.get_stoi()
df3=pd.DataFrame.from_dict([list1])
df3.to_csv(r'D:\fzu\lw\jupyter\pycham2023\amps_predictor\dict_AWLSTM.csv',index=False)

print('------------------------1划分训练与测试集创建train_data和test_data------------------------------')
print('------------------------1划分训练与测试集创建train_data和test_data------------------------------')
#划分训练与测试集创建train_data和test_data

# data=data.drop(['target'],axis=1)
# data=data.drop(['length'],axis=1)
cols = list(data.columns)
data=data[cols[::-1]]# label在前，sequence在后
# data.rename(columns={'review':'label'},inplace=True)
i = int(len(data)*0.8)
train_data = data.sample(i,random_state=2023)#随机取样品
#print(train_data)
train_data.target.value_counts()
print(train_data.target.value_counts())
test_data = data.iloc[data.index[~data.index.isin(train_data.index)]]
print(test_data.target.value_counts())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(train_data.values[0])
print(test_data.values[0])
print(data.head())
print('------------------------2创建Dataloader------------------------------')
print('------------------------2创建Dataloader------------------------------')
print('------------------------2创建Dataloader------------------------------')

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(_label)
        # _text = _text + [0] * (100-len(_text))
        precessed_text = vocab(_text)
        precessed_text = precessed_text + [0] * (70-len(precessed_text))
        precessed_text = torch.tensor(precessed_text, dtype=torch.int64)
        text_list.append(precessed_text)
    label_list = torch.tensor(label_list)
    # text_list = torch.tensor(torch.cat(text_list, dim=0))
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return text_list.to(device), label_list.to(device)

BATCH_SIZE=64

train_dl = DataLoader(train_data.values, batch_size=BATCH_SIZE,
                      collate_fn=collate_batch,
                      shuffle=True)

test_dl = DataLoader(test_data.values, batch_size=BATCH_SIZE,
                       collate_fn=collate_batch,shuffle=True)

# test_data_AMP=test_data[test_data['label']==0]
# test_data_AMP_dl = DataLoader(test_data_AMP.values, batch_size=BATCH_SIZE,
#                        collate_fn=collate_batch)
#
# test_data_nonAMP=test_data[test_data['label']==1]
# test_data_nonAMP_dl = DataLoader(test_data_nonAMP.values, batch_size=BATCH_SIZE,
#                        collate_fn=collate_batch)
#检测批次读取数据及对应标签数量
label_batch, text_batch = next(iter(train_dl))
# print((torch.unique(label_batch,return_counts=True)),text_batch)
# label_batch, text_batch = next(iter(test_dl))
# print((torch.unique(label_batch, return_counts=True)), text_batch)

print('------------------------3定义LSTM模型------------------------------')
print('------------------------3定义LSTM模型------------------------------')
print('------------------------3定义LSTM模型------------------------------')
#定义LSTM模型
print('vocab_size',vocab_size)
vocab_size = len(vocab)
embeding_dim = 300
hidden_size = 70
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
model = LSTM_Net(vocab_size, embeding_dim).to(device).cuda(1)
print(model)
print('------------------------4定义fit训练函数fit------------------------------')
print('------------------------4定义fit训练函数fit------------------------------')

# 定义fit训练函数fit(epoch, model, trainloader, testloader)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.4)
from torch.optim import lr_scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# print('------------------------4定义fit训练函数fit------------------------------')
# print('------------------------4定义fit训练函数fit------------------------------')
#
# # 定义fit训练函数fit(epoch, model, trainloader, testloader)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# from torch.optim import lr_scheduler
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)


def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for x, y in trainloader:
        # print(x.shape)
        x, y = x.to(device).cuda(1), y.to(device).cuda(1)
        y_pred = model(x)
#        print(x.shape)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    #    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device).cuda(1), y.to(device).cuda(1)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

print('------------------------5训练及保持最优模型------------------------------')
print('------------------------5训练及保持最优模型------------------------------')
print('------------------------5训练及保持最优模型------------------------------')
import copy

epochs = 70
train_loss = []
train_acc = []
test_loss = []
test_acc = []

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl,)
    if epoch_test_acc > best_acc:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = epoch_test_acc

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

model.load_state_dict(best_model_wts)

#保存模型方式1
PATH1 = r'D:\fzu\lw\jupyter\pycham2023\amps_predictor\0.8LSTM200AA_model_way1.pth'
torch.save(model,PATH1)
#保存模型方式2
PATH2 = r'D:\fzu\lw\jupyter\pycham2023\amps_predictor\AWLSTM_2.pth'
torch.save(model.state_dict(),PATH2)
#保存模型方式3
# PATH3 = '1/pythonProject4/3-win/2023/result/0.8LSTM200AA_model_way3.pth'
# torch.save(
#     {'epoch':epochs,
#      'model.state_dict':model.state_dict(),
#      'optimizer.state_dict':optimizer.state_dict(),
#     'loss':loss_fn},PATH3)

result = pd.DataFrame()
result['train_acc'] = train_acc
result['test_acc'] = test_acc
result['train_loss'] = train_loss
result['test_loss'] = test_loss
print(result)
result.to_csv(r'D:\fzu\lw\jupyter\pycham2023\amps_predictor\AWLSTM_accloss.csv', index=False)

print('------------------------6画图------------------------------')
#准确率
from matplotlib.pyplot import figure
# figure(figsize=(9.2,7.5))
epochs=epochs
# train_acc = hmp_lstm.train_acc
# test_acc=hmp_lstm.test_acc
#绘图
plt.rcParams['font.serif']=['Times New Roman']
plt.rc('font',family='Times New Roman',size=26)#极其重要
plt.plot(range(epochs), train_acc, c='r', label='Train Accuracy')
plt.plot(range(epochs), test_acc, c='b', label='Test Accuracy')
plt.title('AWLSTM Accuracy',fontsize=26,)
plt.xlabel("Epoch",fontsize=26,) # 横轴fontweight='bold'
plt.ylabel("Accuracy",fontsize=26,) # 纵轴fontweight='bold'
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 27}
plt.legend(prop=font1,loc='best',frameon=False)
#设置坐标轴字体大小
plt.yticks(np.arange(0.80,1.01,0.05),fontproperties='Times New Roman',size=26)#设置大小及加粗
plt.xticks(np.arange(0,epochs+1,epochs/10),fontproperties='Times New Roman', size=26,)
plt.grid(b=None)
ax = plt.gca()#获取边框
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')  # 设置上‘脊梁’黑色
ax.spines['right'].set_color('black')  # 设置上‘脊梁’为无色
ax.spines['left'].set_color('black') 
bwith=1
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.grid(b=None)
plt.savefig(r"D:\fzu\lw\jupyter\pycham2023\amps_predictor\AWLSTM_Accuracy.png",dpi=1200,bbox_inches='tight')
plt.show() # 显示


#Loss
from matplotlib.pyplot import figure
# figure(figsize=(9.2,7.5))
epochs=epochs
# train_acc = hmp_lstm.train_acc
# test_acc=hmp_lstm.test_acc
#绘图
plt.rcParams['font.serif']=['Times New Roman']
plt.rc('font',family='Times New Roman',size=26)#极其重要
plt.plot(range(epochs), train_loss, c='r', label='Train Loss')
plt.plot(range(epochs), test_loss, c='b', label='Test Loss')
plt.title('AWLSTM Loss',fontsize=26,)
plt.xlabel("Epoch",fontsize=26,) # 横轴fontweight='bold'
plt.ylabel("Loss",fontsize=26,) # 纵轴fontweight='bold'
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 27}
plt.legend(prop=font1,loc='best',frameon=False)
#设置坐标轴字体大小
plt.yticks(fontproperties='Times New Roman',size=26)#设置大小及加粗
plt.xticks(np.arange(0,epochs+1,epochs/10),fontproperties='Times New Roman', size=26,)
# plt.grid(b=None)
plt.grid()
ax = plt.gca()#获取边框
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')  # 设置上‘脊梁’黑色
ax.spines['right'].set_color('black')  # 设置上‘脊梁’为无色
ax.spines['left'].set_color('black') 
bwith=1
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.grid(b=None)
plt.savefig(r"D:\fzu\lw\jupyter\pycham2023\amps_predictor\AWLSTM_Loss.png",dpi=1200,bbox_inches='tight')
plt.show() # 显示

print('------------------------罗维先生，你好。训练已经完成，祝你科研顺利，生活愉快------------------------------')
print('------------------------罗维先生，你好。训练已经完成，祝你科研顺利，生活愉快------------------------------')
print('------------------------罗维先生，你好。训练已经完成，祝你科研顺利，生活愉快------------------------------')

# print('训练数据集' + train_data.label.value_counts())
# print('测试数据集' + test_data.label.value_counts())
# print('全AMP测试数据集' + test_data_AMP.label.value_counts())

# print('------------------------7保存模型------------------------------')
# print('------------------------7保存模型-------------------------------')
# print('------------------------7保存模型------------------------------')
# #保存模型
# PATH= './LSTM200AA_model.pth'
# torch.save(model.state_dict(),PATH)

# 调用模型
# LSTM200AA_model = LSTM_Net()
# #恢复权重参数
# PATH = 'LSTM200AA_model.pth'
# LSTM200AA_model.load_state_dict(torch.load(PATH))
# #预测
# model_eval= LSTM200AA_model.eval()
# data_pred = pd.read_csv(r'./12.csv')
# data_pred = data_pred.sequence
# # _, x_pred =next(iter(test_dl)
# y_pred = model_eval(data_pred)
# print(y_pred)
# data_pred.target=y_pred
# to.csv(r"  csv")


#ctri+/
# def yield_tokens(data):
#     for text in data:
#         yield text
# from torchtext.vocab import build_vocab_from_iterator
# vocab = build_vocab_from_iterator(yield_tokens(data_pred.sequence))
# precessed_text = torch.tensor(vocab(x) for x in data_pred.sequence), dtype=torch.int64)
# precessed_text = precessed_text + [0] * (200 - len(precessed_text))
# precessed_text = torch.tensor(precessed_text, dtype=torch.int64)

