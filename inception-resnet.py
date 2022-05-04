import torch
import torch.nn as nn

'''
Implementing the Inception GoogLeNet arcitecture from https://arxiv.org/pdf/1409.4842.pdf

'''

class inceptNet(nn.Module):
    '''
    Channels and parameters pulled from Table 1 of paper
    '''

    def __init__(self, aux = True, classes=10):
        super(inceptNet, self).__init__()
        assert aux == True or aux == False
        self.aux_logits = aux
        self.conv1 = convolutional_set(in_chans=3, out_chans=64, kernel_size=(7, 7),
                                       stride=(2, 2), padding=(3, 3))
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = convolutional_set(64, 192, kernel_size=3, stride=1, padding=1)
        # use maxpool here again

        # order: in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, out_pool_1x1
        self.inception3a = inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
        # use maxpool here again
        self.inception4a = inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception(512, 112, 114, 288, 32, 64, 64)
        self.inception4e = inception(528, 256, 160, 320, 32, 128, 128)
        # maxpool here
        self.inception5a = inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception(832, 384, 192, 384, 48, 128, 128)
        self.averagepool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, classes)

        if self.aux_logits:
            self.aux1 = inceptAux(512, classes)
            self.aux2 = inceptAux(528, classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool2(x)

        x = self.inception4a(x)
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.max_pool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.averagepool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x


class inception(nn.Module):
    '''
    Inception block described in Figure 2(b) of the linked paper
    '''

    def __init__(self, in_chans, out_1x1, reduct_3x3, out_3x3, reduct_5x5, out_5x5, out_pool_1x1):
        super(inception, self).__init__()

        # creating branches of inception block
        self.b1 = convolutional_set(in_chans, out_1x1, kernel_size=(1,1))

        self.b2 = nn.Sequential(
            convolutional_set(in_chans, reduct_3x3, kernel_size=(1,1)),
            convolutional_set(reduct_3x3, out_3x3, kernel_size=(3,3), padding=(1,1)),  # kernel=3, stride = 1, (W−F+2P)/S+1
        )

        self.b3 = nn.Sequential(
            convolutional_set(in_chans, reduct_5x5, kernel_size=(1,1)),  # padding=0, stride=1
            convolutional_set(reduct_5x5, out_5x5, kernel_size=(5,5), padding=(2,2)),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            convolutional_set(in_chans, out_pool_1x1, kernel_size=(1,1)),
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)

class inceptAux(nn.Module):
    def __init__(self, in_chans, classes):
        super(inceptAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = convolutional_set(in_chans, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024, classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0],-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class convolutional_set(nn.Module):

    def __init__(self, in_chans, out_chans, **kwargs):
        super(convolutional_set, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_chans, out_chans, **kwargs)

    def foward(self, x):
        return self.relu(self.conv(x))


# testing sizes. We want 3 images of 1000
if __name__ == '__main__':
    x = torch.randn(3, 3, 224, 224)
    model = inceptNet(aux = True, classes = 10)
    print(model(x).shape)
