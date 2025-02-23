import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from RCRI_CM import rcri_cm
from RCRI_CM import compute_LOA


class get_model(nn.Module):
    def __init__(self,num_class, With_normal=False):
        super(get_model, self).__init__()
        channel_input = 64
        self.With_normal = With_normal
        self.moudle1 = rcri_cm(256 , 8   , 0   + channel_input, [32] ,  global_cm=False)
        self.moudle2 = rcri_cm(128 , 16  , 32  + channel_input, [64] ,  global_cm=False)
        self.moudle3 = rcri_cm(64  , 32  , 64  + channel_input, [128],  global_cm=False)
        self.moudle4 = rcri_cm(32  , 32  , 128 + channel_input, [256],  global_cm=False)
        self.moudle5 = rcri_cm(None, None, 256 + channel_input, [512],  global_cm=True )
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.With_normal:
            LOA = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]
        else:
            LOA = compute_LOA(xyz)
        C1, LOA_1, F1 = self.moudle1(xyz, LOA, None)
        C2, LOA_2, F2 = self.moudle2(C1, LOA_1, F1)
        C3, LOA_3, F3 = self.moudle3(C2, LOA_2, F2)
        C4, LOA_4, F4 = self.moudle4(C3, LOA_3, F3)
        C5, LOA_5, F5 = self.moudle5(C4, LOA_4, F4)
        F_out = F5.view(B, 512)
        F_out = self.drop1(F.relu(self.bn1(self.fc1(F_out))))
        F_out = self.drop2(F.relu(self.bn2(self.fc2(F_out))))
        F_out = self.fc3(F_out)
        F_out = F.log_softmax(F_out, -1)

        return F_out,F5


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss
    
class CustomDataset(Dataset):
    def __init__(self, data, train):
        self.data = data
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.train[idx]