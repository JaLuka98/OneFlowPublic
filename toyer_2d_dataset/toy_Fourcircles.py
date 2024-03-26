# Standard imports
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import zuko
import json
import os

# Importing other scripts
import datasets_generation as data_gen
import training_utils

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def main():

    # generating "data" (FourCircles) and "simulation" (CheckerBoard) training samples
    data       = data_gen.FourCircles( 1000000 )
    simulation = data_gen.CheckerboardDataset( 1000000 )

    # I had to do it so it is acceptable by pytorch
    data       = np.concatenate( [np.array(data[:,0]).reshape(-1,1), np.array(data[:,1]).reshape(-1,1)] , axis = 1)
    simulation = np.concatenate( [np.array(simulation[:,0]).reshape(-1,1),np.array(simulation[:,1]).reshape(-1,1)], axis = 1)

    # Creating the conditions tensors True for data and False for simulation
    data_conditions       = np.ones_like(data[:,0]).reshape(-1,1)
    simulation_conditions = (np.ones_like(simulation[:,0])*0).reshape(-1,1)

    # Combine real data and simulated data
    combined_data = np.concatenate([data, simulation], axis=0)
    combined_conditions = np.concatenate([data_conditions, simulation_conditions], axis=0)

    # Split into training+validation (70%) and test (30%)
    train_val_data, test_data, train_val_conditions, test_conditions = train_test_split(
        combined_data, combined_conditions, test_size=0.3, random_state=42)

    # Split training+validation into training (60/70 of the original) and validation (10/70 of the original)
    train_data, val_data, train_conditions, val_conditions = train_test_split(
        train_val_data, train_val_conditions, test_size=0.1/0.7, random_state=42)
 
    # From numpy to pytorch tensors
    training_inputs = torch.Tensor(train_data)
    validation_inputs = torch.Tensor(val_data)
    test_inputs = torch.Tensor(test_data)
    
    training_conditions = torch.Tensor(train_conditions)
    validation_conditions = torch.Tensor(val_conditions)
    test_conditions = torch.Tensor(test_conditions)

    # The test conditions will not be used here, only in the plotting part! 
    # Lets save them to use after the plotting part
    torch.save(test_inputs, 'Four_circles_inputs.pt')
    torch.save(test_conditions, 'Four_circles_conditions.pt')

    # Now, lets set up the flow model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lets read the .json containing the flow information
    with open('flow_configuration.json', 'r') as f:
        config = json.load(f)
    flow = zuko.flows.NSF( training_inputs.size()[1] , training_conditions.size()[1] , transforms=config['n_transforms'], bins=config['n_spline_bins'] ,hidden_features=[config['aux_nodes']] * config['aux_layers'])

    # Train to maximize the log-likelihood
    lr = 1e-3
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    early_stopper = training_utils.EarlyStopper(patience = 10, min_delta=0.000)
    
    batch_size = config['batch_size']
    n_batches,n_epochs = int( len(training_inputs)/batch_size ), 125
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_batches, eta_min= 1e-10) #Cossine anealing lr

    # everything to GPU!
    flow = flow.to(device)
    training_inputs     = training_inputs.to(device)
    training_conditions = training_conditions.to(device)
    val_loss_array,train_loss = [],[]

    dataset = training_utils.CustomDataset(training_inputs, training_conditions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # before the training starts lets create the directories where the flow states will be stored
    os.makedirs( "./saved_models", exist_ok=True )

    print(f'{Color.GREEN}Starting training!{Color.RESET}')

    # Basic trainig loop
    for epoch in range(n_epochs):
        
        for inputs, conditions in dataloader:
            
            loss = -flow( conditions ).log_prob( inputs )  
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            scheduler.step() # cossine anealing is updated at each batch, not epoch! (https://arxiv.org/pdf/1608.03983.pdf)
            # More details of its implementation -> https://discuss.pytorch.org/t/how-to-implement-torch-optim-lr-scheduler-cosineannealinglr/28797/6

        with torch.no_grad():
            
            val_loss = -flow( validation_conditions.to(device) ).log_prob( validation_inputs.to(device) )
            val_loss_array.append( float(val_loss.mean()) )
            
            # Appending losses to its respective arrays
            train_loss.append( float( loss ) )

            # Restarting the leraning rate to 1e-3 every epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_batches) 
            
            print(f"\033[1;34mEpoch: {epoch}\033[0m, \033[1;32mTraining Loss: {float( loss.mean() ):.3f}\033[0m, \033[1;31mValidation Loss: {float(val_loss.mean()):.3f}\033[0m")

            #plotting the loss function of the training
            training_utils.loss_plot(val_loss_array, train_loss, model='Four_circles')

            # Saving the model of each epoch
            torch.save(flow.state_dict() , 'saved_models/model_' + str(epoch) + '.pth' )
            if(early_stopper.early_stop(float( val_loss.mean() )) ):
                print( '\nEarly stop criterion achieved at epoch: ', epoch )
                break

    print(f'\n{Color.GREEN}Training ended! ============================================{Color.RESET}')
    print(f'{Color.CYAN}Final Validation loss:{Color.RESET} {val_loss.mean():.4f}')
    print(f'{Color.MAGENTA}Lowest validation loss occurred at epoch:{Color.RESET} {np.argmin(np.array(val_loss_array))} {Color.YELLOW}value:{Color.RESET} {np.min(np.array(val_loss_array)):.4f}') 

    #now that the training is ended, lets load the model with the lowest validation loss
    flow.load_state_dict(torch.load('saved_models/model_' +  str(np.argmin( np.array( val_loss_array ) )) +'.pth'  ))
    torch.save(flow.state_dict() , 'model_fourcircles.pth' )

if __name__ == "__main__":
    main()

