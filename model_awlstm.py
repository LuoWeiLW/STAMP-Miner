# core/model_awlstm.py
import torch.nn as nn
import torch.nn.functional as F

class AWLSTM_Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_size=70, max_len=70):
        super(AWLSTM_Net, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * max_len, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # x shape: (batch, max_len)
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.contiguous().view(len(x), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.8)
        x = F.dropout(F.relu(self.fc2(x)), p=0.4)
        x = self.fc3(x)
        return x