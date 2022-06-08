import torch
from network_model import NeuralNetwork
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

with open("mnist.pkl","rb") as f:
    mnist = pickle.load(f)
X,y = mnist["data"],mnist["target"].astype(int)

X = torch.from_numpy(X.values)
y = torch.from_numpy(y.values).int()
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.1,random_state=42)


# Create data-loader to make it more easy to train

class Data(Dataset):
    """
    """
    def __init__(self,X,y):
        self.X,self.y = X,y

    def __getitem__(self, i):
        return self.X[i],self.y[i]

    def __len__(self):
        return self.X.shape[0]




#%%

# Create instance
net = NeuralNetwork(dropout=0.5)
device = "cuda" if torch.cuda.is_available() else "cpu"  #Use GPU if available else CPU
net = net.to(device)
X_train = X_train.to(device)
X_val = X_val.to(device)

y_train = y_train.to(device)
y_val = y_val.to(device)

N_EPOCHS = 50 # Number of times the neural network see the entire dataset
BATCH_SIZE = 256 # Number of instances it sees pr iteraion

# Create dataloader
data = Data(X_train, y_train)
dl = DataLoader(data, batch_size=BATCH_SIZE,shuffle=True, num_workers=0)



# Define loss and score

loss = CrossEntropyLoss()

def score(pred,target): #Accuracy
    pred_class = pred.argmax(dim=1)
    return (pred_class == target).sum()/len(target)



# Define optimizer-algorithm
optimizer = torch.optim.Adam(params = net.parameters(),lr = 0.001)
scheduler = ReduceLROnPlateau(optimizer,mode="min",patience = 5,verbose=True)

TRAIN_LOSS = np.zeros(N_EPOCHS)
VAL_LOSS = np.zeros(N_EPOCHS)
TRAIN_SCORE = np.zeros(N_EPOCHS)
VAL_SCORE = np.zeros(N_EPOCHS)


#### Train the netowrk ####
print(f"Epoch     Train-loss    Val-loss    Train-score    Val-score")
print("----------------------------------------------------------------")
for e in range(0,N_EPOCHS):

    net.train() # Set network in train-mode
    for X_batch,y_batch in dl:

        # Iterate
        optimizer.zero_grad()
        pred = net(X_batch.float())
        loss_batch = loss(pred,y_batch.long())

        loss_batch.backward() # Take a step
        optimizer.step()

    # Evaluate
    net.eval()

    # Train loss/score
    train_pred = net(X_train.float())
    train_loss = loss(train_pred,y_train.long())
    train_score = score(train_pred,y_train)

    # Validateion loss/score
    val_pred = net(X_val.float())
    val_loss = loss(val_pred,y_val.long())
    val_score = score(val_pred,y_val)
    scheduler.step(val_loss)


    TRAIN_LOSS[e] = train_loss
    TRAIN_SCORE[e] = train_score

    VAL_LOSS[e] = val_loss
    VAL_SCORE[e] = val_score

    print(f"{e+1}/{N_EPOCHS}    {train_loss:2.2f}           {val_loss:2.2f}           {train_score:2.2f}            {val_score:2.2f}")




#%%

