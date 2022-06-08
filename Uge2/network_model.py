from torch import nn

class NeuralNetwork(nn.Module):

    def __init__(self,dropout):

        super(NeuralNetwork,self).__init__()

        self.l0 = nn.Linear(784,512)
        self.l1 = nn.Linear(self.l0.out_features,256)
        self.l2 = nn.Linear(self.l1.out_features,10)

        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.ReLU(inplace = False)

        self.dropout = nn.Dropout(dropout)



    def forward(self, x):

        # Dense layer
        x = self.l0(x)
        x = self.activation(x)
        x = self.dropout(x)


        x = self.l1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.l2(x)
        x = self.dropout(x)


        return x


