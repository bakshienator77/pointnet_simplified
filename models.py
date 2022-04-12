import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # pass
        self.features_local = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.features_global = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        # self.maxpool = nn.Max
        # torch.max(x, 2, keepdim=True)[0]

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # pass
        x = points.transpose(2, 1)
        # print("shape is now (B, 3, N)?: ", x.shape)
        #local embeddings
        x = self.features_local(x)
        # print("Shape should now be (B, 64, N): ", x.shape)
        x = self.features_global(x)
        # print("Shape should now be (B, 1024, N): ", x.shape)
        x = torch.max(x, 2)[0]
        # print("Shape should now be (B, 1024): ", x.shape)
        x = self.fc(x)
        # print("Shape should now be (B, 3): ", x.shape)
        return x




# ------ TO DO ------
class seg_model(cls_model):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        # pass
        self.features_points = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Conv1d(128, num_seg_classes, 1),
            # nn.BatchNorm1d(num_seg_classes),
            # nn.ReLU(),
        )
    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        # pass
        x = points.transpose(2, 1)
        # print("shape is now (B, 3, N)?: ", x.shape)
        #local embeddings
        x = self.features_local(x)
        # print("Shape should now be (B, 64, N): ", x.shape)
        x_global = self.features_global(x)
        # print("Shape should now be (B, 1024, N): ", x_global.shape)
        x_global = torch.max(x_global, 2, keepdim=True)[0].repeat((1,1,x.shape[-1]))
        # print("Shape should now be (B, 1024, N): ", x_global.shape)
        x = torch.cat([x, x_global], dim=1)
        # print("Shape should now be (B, 1088, N): ", x.shape)
        x = self.features_points(x)
        # print("Shape should now be (B, 128, N): ", x.shape)
        x = self.output(x).transpose(2, 1)
        # print("Shape should now be (B, N, 6): ", x.shape)
        return x



