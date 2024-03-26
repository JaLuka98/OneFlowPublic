# Scripts usefull during training!
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# For data loader
class CustomDataset(Dataset):
    def __init__(self, inputs, conditions):
        self.inputs = inputs
        self.conditions = conditions

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.conditions[idx]

# simple plot to keep track of the loss
def loss_plot(val_loss_array, train_loss_array, model):
    plt.figure(figsize=(8, 6))
    plt.plot( val_loss_array,   linewidth = 3, label = 'Validation loss' )
    plt.plot( train_loss_array, linewidth = 3,  label = 'Training loss' )
    
    # Lets set the limit based on the values of the training loss
    #plt.ylim( 0.95*np.min(train_loss_array), 1.01*np.max(train_loss_array) )
    plt.legend()
    plt.savefig( 'loss_curve_'+str(model)+'.png' )
    plt.close()

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False