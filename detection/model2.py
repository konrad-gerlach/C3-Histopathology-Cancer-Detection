from torch import nn

class Classifier(nn.Module):
    def __init__(self,img_shape,normalize=True):
        super(Classifier, self).__init__()
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