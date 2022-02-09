import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, w, num_class=36):
        self.X = []
        self.y = []
        self.idx = []
        self.label = []
        self.num_class = num_class

        BlockId = data['BlockId'].values
        Label = data['Label'].values
        LogKeySeq = data['LogKeySeq'].values

        for i, seq in enumerate(LogKeySeq):
            seq = list(map(int, seq.split(' ')))
            seq_one_hot = np.eye(self.num_class)[seq]
            if len(seq) - w < 1:
                self.X.append(np.vstack((np.zeros((w-len(seq)+1, num_class)), seq_one_hot[:-1])))
                self.y.append(seq[-1])

                self.idx.append(BlockId[i])
                self.label.append(Label[i])
            else:
                for j in range(len(seq) - w):
                    self.X.append(seq_one_hot[j:j + w])
                    self.y.append(seq[j + w])

                    self.idx.append(BlockId[i])
                    self.label.append(Label[i])

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        idx = self.idx[index]
        label = self.label[index]

        return X, y, idx, label

    def __len__(self):
        return len(self.X)
