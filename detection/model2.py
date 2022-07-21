from torch import nn

class Big_Konrad(nn.Module):
    def __init__(self,img_shape,normalize=True):
        super(Big_Konrad, self).__init__()
        self.normalize = normalize
        self.img_shape = img_shape
        self.layers = self.get_layers()

    def get_layers(self):
        return nn.Sequential(
            nn.Conv2d(self.img_shape[0],128,kernel_size=7,padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(128,256,kernel_size=5,padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(256,512,kernel_size=3,padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(512,64,kernel_size=1,padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Flatten(),
            nn.Linear(int(64*self.img_shape[1]/16*self.img_shape[2]/16), 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200, 400),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(400, 1)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

# inspired by: https://blog.ineuron.ai/AlexNet-CNN-architecture-With-Implementation-in-Keras-Q4strWr4iZ
class Alex_Net(nn.Module):
    def __init__(self):
        super(Alex_Net, self).__init__()
        self.layers = self.build_layers()

    def build_layers(self):
        return nn.Sequential(
            nn.Conv2d(3,96,kernel_size=7,stride=4,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(96,256,kernel_size=5,padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(384,384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
    
    def get_layers(self):
        return "AlexNet"

    def forward(self, x):
        logits = self.layers(x)
        return logits

class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.layers = self.build_layers()

    def build_layers(self):
        return nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding="same"),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(64,128,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding="same"),
            nn.ReLU(),         
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(128,256,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding="same"),
            nn.ReLU(),         
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(256,512,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding="same"),
            nn.ReLU(),         
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(512,512,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding="same"),
            nn.ReLU(),         
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

           
            nn.Flatten(),
            nn.Linear(4608, 4608),
            nn.ReLU(),
            nn.Linear(4608, 1)
        )
    
    def get_layers(self):
        return "VGG16"

    def forward(self, x):
        logits = self.layers(x)
        return logits


class No_Conv(nn.Module):
    def __init__(self):
        super(No_Conv, self).__init__()
        self.layers = self.build_layers()

    def build_layers(self):
        return nn.Sequential(           
            nn.Flatten(),
            nn.Linear(27648, 9216),
            nn.ReLU(),
            nn.Linear(9216, 1)
        )
    
    def get_layers(self):
        return "No_Conv"

    def forward(self, x):
        logits = self.layers(x)
        return logits