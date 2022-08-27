from torch import nn
from abc import abstractmethod


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.ModuleList(self.get_layers())

    @abstractmethod
    def get_layers(self):
        pass

    def forward(self, x):
        logits = nn.Sequential(*self.layers)(x)
        return logits
    
    def forward_per_layer(self,x):
        outputs = []
        previous_layer_output = x
        for layer in self.layers:
            previous_layer_output = layer(previous_layer_output)
            outputs.append(previous_layer_output)
        return outputs

# our final model
class Big_Konrad(Model):
    def __init__(self, fc_layer_size, conv_dropout, fully_dropout):
        super(Model, self).__init__()
        print("The model in use: ", self.__class__.__name__)
        self.layers = nn.ModuleList(self.get_layers(s=fc_layer_size, c=conv_dropout, f=fully_dropout))

    def get_layers(self, s, c, f):
        return [
            # 1st convolutional layer
            nn.Dropout2d(p=c),
            nn.Conv2d(3, 128, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 2nd convolutional layer
            nn.Dropout2d(p=c),
            nn.Conv2d(128, 256, kernel_size=5, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            # 3rd convolutional layer
            nn.Dropout2d(p=c),
            nn.Conv2d(256, 512, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 4th convolutional layer
            nn.Dropout2d(p=c),
            nn.Conv2d(512, 64, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # fully connected layers
            nn.Flatten(),
            nn.Dropout(p=f),
            nn.Linear(int(64 * 96 / 16 * 96 / 16), s),
            nn.BatchNorm1d(s),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=f),
            nn.Linear(s, 2 * s),
            nn.BatchNorm1d(2 * s),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=f),
            nn.Linear(2*s, 1)
        ]

# Other basic models we experimented with:
class LeNet(Model):

    def get_layers(self):
        return [
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(7056, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1)
        ]


class Alex_Net(Model):

    def get_layers(self):
        return [
            nn.Conv2d(3,96,kernel_size=7,stride=4,padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        ]

if __name__ == "__main__":
    # show our model
    model = Big_Konrad(200, 0, 0.5)
    print(list(model.modules()))
