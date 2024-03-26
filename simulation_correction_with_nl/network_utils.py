import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import mplhep, hist
plt.style.use([mplhep.style.CMS])
import mplhep as hep
import zuko
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Importing the plotting script
import plotter 

#global number generators seeds
np.random.seed(42)
torch.manual_seed(42)

# Early stopping class: Stop training if validaiton loss does not improve within a number of epochs
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

# For data loader
class CustomDataset(Dataset):
    def __init__(self, inputs, conditions, weights):
        self.inputs = inputs
        self.conditions = conditions
        self.weights = weights

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.conditions[idx], self.weights[idx]

# Class responsable to perform the transformation of the discontinuous variables
class Scale_and_smooth:
    
    def __init__(self, tensor, shift, shift_tail = False):

        self.shift = shift  # max x value of the triangular distribution
        self.tensor = tensor
        self.before_transform = tensor

        # The 'self.shift_tail' is the amout that the tail of the discontinious distributions will be shifted towards the peak at zero.
        # This is also a hyperparameter of the problem, Altough, has not shown big diferences 
        self.shift_tail = 0.9*torch.min( tensor[ tensor > 0 ] )

        self.device = torch.device('cpu')

    def transform(self, dummy):

        # Elements of the tensor to be transformed!
        self.tensor[ self.tensor > 0 ]  = self.tensor[ self.tensor > 0 ] +  self.shift - self.shift_tail 
        self.tensor[ self.tensor == 0 ] = torch.tensor(np.random.triangular( left = 0. , mode = 0, right = self.shift, size = self.tensor[ self.tensor == 0 ].size()[0]   ), dtype = self.tensor.dtype ).to(self.device)  

        self.tensor = torch.log( self.tensor + 1e-3)

        return self.tensor

    def invert_transform(self, tensor,processed = False):

        # Inverting the transformation
        tensor = torch.exp( tensor) - 1e-3
        tensor[ tensor < self.shift ] = 0.0
        tensor[ tensor >= self.shift ]  = tensor[ tensor >= self.shift ]  + self.shift_tail - self.shift
        
        return tensor

# This class perfoms the heavy work, it reads treat the data, train and validate the flow  
class Toy_data_framework:
    
    # Read the datasets and sets the conditons and inputs variables that will be used in training
    def __init__(self, data_path, configuration, n_transforms, n_splines_bins, aux_nodes, aux_layers, shift_discontinious_var1, shift_discontinious_var2, max_epoch_number, initial_lr, batch_size, Is_Autoregressive):

        # Flag to select learning rate scheduler
        self.cossine_annealing_scheduler = True

        print('Begin of data reading and pre-processing.\n')

        # Reading toy simulation and data .csv files
        self.files_path = data_path
        self.data       = pd.read_csv(self.files_path + "my_data_withNoise.csv")
        self.simulation = pd.read_csv(self.files_path + "my_MC_withNoise.csv")

        # The model doesnt need a gigantic number of events, so I am clipping them
        self.data = self.data[:2500000]
        self.simulation = self.simulation[:2500000]

        #Names of the conditions and inputs
        self.conditions_list = ["pT","eta","noise"]      # Noise should be something to simulate pile-up to pile-up
        self.inputs          = ["v1A","v2A","v1B","v2B"] #lets do some tests to check if there is any problematic variables

        # Autoregressive or coupling blocks flows
        self.Is_Autoregressive = Is_Autoregressive

        # Setting the device
        self.device = torch.device('cpu') 

        self.configuration =  configuration
        try:
            #print('\nThis run dump folder: ', os.getcwd() + '/results/' +self.configuration + '/')
            os.makedirs(os.getcwd() + '/results/' +self.configuration + '/',  exist_ok=True)
            os.makedirs(os.getcwd() + '/results/' +self.configuration + '/saved_states/',  exist_ok=True)
        except:
            print('\nIt was not possible to open the dump folder')
            exit()

        self.dump_folder = os.getcwd() + '/results/' +self.configuration + '/'

        # Defining the flow hyperparameters
        self.n_transforms   = n_transforms
        self.n_splines_bins = n_splines_bins
        self.aux_nodes      = aux_nodes
        self.aux_layers     = aux_layers

        # discontinuous transformation hyperparameters
        self.shift_discontinious_var1 = shift_discontinious_var1   
        self.shift_discontinious_var2 = shift_discontinious_var2

        # general training hyperparameters
        self.max_epoch_number = max_epoch_number
        self.initial_lr       = initial_lr
        self.batch_size       = batch_size

        # Opening a .txt file to store the hyperparamets used in the training of the flow
        txt_file  = open( self.dump_folder + 'network_information.txt', 'w')

        txt_file.write('\nFlow hyperparameters:\n\n')
        txt_file.write('Number of transfomations: '             + str( self.n_transforms ) + '\n' )
        txt_file.write('Number of spline bins: '                + str( self.n_splines_bins ) + '\n' )
        txt_file.write('Number of aux nodes: '                  + str( self.aux_nodes ) + '\n' )
        txt_file.write('Number of aux layers: '                 + str( self.aux_layers  ) + '\n' )
        txt_file.write('\nNdiscontinuous 01 shift: '            + str( self.shift_discontinious_var1 ) + '\n' )
        txt_file.write('\nNdiscontinuous 02 shift: '            + str( self.shift_discontinious_var2 ) + '\n' )
        txt_file.close()

    # Second processing step - uses dataframes and separates its contents into inputs and conditions for the data and simulation
    def treat_data(self):
      
        self.data_inputs         = torch.tensor(np.array(self.data[self.inputs]))
        self.simulation_inputs   = torch.tensor(np.array(self.simulation[self.inputs]))

        # Creating the tensors to store the conditions + the IsData boolean (1 for data 0 for simulation)
        self.data_conditions       = torch.cat( [ torch.tensor(np.array(self.data[ self.conditions_list ]))       ,   torch.tensor( np.ones_like( self.data_inputs[:,0] ) ).view(-1,1)], axis = 1)
        self.simulation_conditions = torch.cat( [ torch.tensor(np.array(self.simulation[ self.conditions_list ])) ,   torch.tensor( 0*np.ones_like( self.simulation_inputs[:,0] ) ).view(-1,1)], axis = 1)

        # We reweight the toy simulation conditions to match the data conditions, and hence "eliminate" kinematics effects
        self.perform_reweighting()

        # Shiffiting tensors to  choosen device
        self.data_inputs           = self.data_inputs.to(self.device)
        self.simulation_inputs     = self.simulation_inputs.to(self.device)
        self.data_conditions       = self.data_conditions.to(self.device)
        self.simulation_conditions = self.simulation_conditions.to(self.device)

        # Creating tensors to store the inputs/conditions before the transformations
        self.simulation_inputs_before_transformations = self.simulation_inputs.detach().clone()
        self.data_inputs_before_transformation = self.data_inputs.detach().clone()

        # Performing the discontinuous variables transformations
        self.discontinous_variables_transformation()

        # Separating the events into training, validation and test datasets
        training_inputs_dataset     = np.concatenate([self.data_inputs     , self.simulation_inputs])
        training_conditions_dataset = np.concatenate([self.data_conditions , self.simulation_conditions])
        training_weights_dataset    = np.concatenate([self.data_weights    , self.simulation_weights])

        # Initial split: Separate out a test set (60 for training, 10 for validation and 30 for test)
        X_train_val, X_test, cond_train_val, cond_test, weights_train_val, weights_test = train_test_split(
            training_inputs_dataset, training_conditions_dataset, training_weights_dataset, 
            test_size=0.3, random_state=42)

        # Second split: Divide the remaining data into training and validation sets
        X_train, X_val, cond_train, cond_val, weights_train, weights_val = train_test_split(
            X_train_val, cond_train_val, weights_train_val, 
            test_size=0.1/0.7, random_state=42)

        # To pytorch tensors
        self.training_inputs, self.training_conditions, self.training_weights = torch.tensor(X_train),torch.tensor(cond_train),torch.tensor(weights_train)
        self.validation_inputs, self.validation_conditions, self.validation_weights = torch.tensor(X_val),torch.tensor(cond_val),torch.tensor(weights_val)
        self.test_inputs, self.test_conditions, self.test_weights = torch.tensor(X_test),torch.tensor(cond_test),torch.tensor(weights_test)
        
        # Asserting that same type tensors have the same size
        assert len( self.training_inputs )   == len( self.training_conditions )
        assert len( self.validation_inputs ) == len( self.validation_conditions )
        
        assert len( self.training_weights ) == len( self.training_inputs )
        assert len( self.validation_weights ) == len( self.validation_inputs )

        assert len( self.test_weights ) == len( self.test_inputs )
        assert len( self.test_weights ) == len( self.test_inputs )

        self.standardize()
        
        # Making plots of after and before transformations!
        self.plot_transformation_before_and_after()

    # Due to the diferences in the kinematic distirbutions of the data and MC a reweithing must be performed to account for this
    def perform_reweighting(self):
        
        """
        Reweights a 3D histogram of MC (data1) to match the 3D histogram of Data (data2).
        
        Parameters:
        - data1, data2: Arrays of 3D data points.
        - weights1, weights2: Arrays of weights for data1 and data2.
        - bins: Number of bins or bin edges for each dimension.
        
        Returns:
        - reweighted_weights1: The reweighted weights for data1.
        """

        mc_weights = np.ones_like( self.simulation_conditions[:,0] )
        mc_weights = mc_weights/np.sum( mc_weights )

        data_weights = np.ones_like( self.data_conditions [:,0] )
        self.data_weights = torch.tensor(data_weights/np.sum( data_weights ))

        # Definning the reweigthing binning! - Bins were chossen such as each bin has ~ the same number of events
        pt_bins = [0, 0.0573455764104733, 0.11735780800697526, 0.18102048016108208, 0.2493241033395276, 0.3228878774665439, 0.4023036301626174, 0.4879565901202798, 0.5828565113995026, 0.6871772394764465, 0.8041574902442367, 0.9375624166892256, 1.089764326063702, 1.2710573139239396, 1.4948638970346209, 1.7809796071603032, 2.1825766970148663, 2.8658669583416736, 3.6, 100]
        eta_bins = [0, 0.05, 0.103424993074548, 0.20632838071077952, 0.3099353025917125, 0.413059935425879, 0.5170050025148473, 0.6211492864979111, 0.7257311325988597, 0.8316217193991894, 0.9389161117512556, 1.0481460106026568, 1.1591517153192457, 1.274699791172384, 1.396448622766261, 1.527009082627225, 1.6745750468877685, 1.8520666111330533, 2.10073835633074, 2.4, 3.208098416554188, 40]
        noise_bins = np.linspace( 0, 3.0,14 )

        bins = [ pt_bins , eta_bins, noise_bins ]

        # Calculate 3D histograms
        data1 = [ np.array(self.simulation_conditions[:,0]) ,   np.array(self.simulation_conditions[:,1]), np.array(self.simulation_conditions[:,2]) ]
        data2 = [ np.array(self.data_conditions[:,0]) ,   np.array(self.data_conditions[:,1]), np.array(self.data_conditions[:,2]) ]

        hist1, edges = np.histogramdd(data1, bins=bins, weights=mc_weights, density=True)
        hist2, _ = np.histogramdd(data2, bins=edges   , weights=data_weights, density=True)

        # Compute reweighting factors
        reweight_factors = np.divide(hist2, hist1, out=np.zeros_like(hist1), where=hist1!=0)

        # Find bin indices for each point in data1
        bin_indices = np.vstack([np.digitize(data1[i], bins=edges[i])-1 for i in range(3)]).T

        # Ensure bin indices are within valid range
        for i in range(3):
            bin_indices[:,i] = np.clip(bin_indices[:,i], 0, len(edges[i]) )
        
        # Apply reweighting factors
        self.simulation_weights = torch.tensor(mc_weights * reweight_factors[bin_indices[:,0], bin_indices[:,1], bin_indices[:,2]])

        # normalizing both to one!
        self.data_weights       = self.data_weights/torch.sum( self.data_weights )
        self.simulation_weights = self.simulation_weights/torch.sum( self.simulation_weights )

    # Standardize the inputs and conditions
    def standardize(self):

        # Calculating and storing the means and standart deviations of the inputs and conditions  
        self.input_means = torch.mean( self.training_inputs, 0 )
        self.input_std   = torch.std(  self.training_inputs, 0 )

        # Since the last entry of the conditions is a boolean, it is not standardized
        self.condition_means = torch.mean( self.training_conditions[:,:-1], 0 )
        self.condition_std   = torch.std(  self.training_conditions[:,:-1], 0 )

        # Standadizing the tensors
        self.training_inputs      = (self.training_inputs - self.input_means )/self.input_std
        self.validation_inputs    = (self.validation_inputs - self.input_means )/self.input_std
        self.test_inputs          = (self.test_inputs - self.input_means )/self.input_std
        self.simulation_inputs    = (self.simulation_inputs - self.input_means )/self.input_std

        self.training_conditions[:,:-1]     = (self.training_conditions[:,:-1] - self.condition_means )/self.condition_std
        self.validation_conditions[:,:-1]   = (self.validation_conditions[:,:-1] - self.condition_means )/self.condition_std
        self.test_conditions[:,:-1]         = (self.test_conditions[:,:-1] - self.condition_means )/self.condition_std
        self.simulation_conditions[:,:-1]   = (self.simulation_conditions[:,:-1] - self.condition_means )/self.condition_std        

    # Invert the standartization operation 
    def invert_standartization(self):
        
        # de-standartization of the tensors
        self.simulation_inputs   = self.simulation_inputs*self.input_std + self.input_means 
        self.validation_inputs   = self.validation_inputs*self.input_std + self.input_means
        self.test_inputs         = self.test_inputs*self.input_std + self.input_means
        self.samples             = self.samples*self.input_std + self.input_means 

        # invert standartization in the conditions
        self.simulation_conditions[:,:-1]  = self.simulation_conditions[:,:-1]*self.condition_std  + self.condition_means
        self.validation_conditions[:,:-1]  = self.validation_conditions[:,:-1]*self.condition_std  + self.condition_means
        self.test_conditions[:,:-1]  = self.test_conditions[:,:-1]*self.condition_std  + self.condition_means

    def discontinous_variables_transformation(self):

        # First transform the simulation variables
        self.transform_sim_var_2 = Scale_and_smooth(self.simulation_inputs[:,2], self.shift_discontinious_var1)
        self.transform_sim_var_3 = Scale_and_smooth(self.simulation_inputs[:,3], self.shift_discontinious_var2)

        self.simulation_inputs[:,2] = self.transform_sim_var_2.transform( self.simulation_inputs[:,2] )
        self.simulation_inputs[:,3] = self.transform_sim_var_3.transform( self.simulation_inputs[:,3] )            

        # Now the data variables
        self.transform_data_var_2 = Scale_and_smooth(self.data_inputs[:,2], self.shift_discontinious_var1)
        self.transform_data_var_3 = Scale_and_smooth(self.data_inputs[:,3], self.shift_discontinious_var2)

        self.data_inputs[:,2] = self.transform_data_var_2.transform( self.data_inputs[:,2] )
        self.data_inputs[:,3] = self.transform_data_var_3.transform( self.data_inputs[:,3] )  

    def discontinous_variables_inverse_transformation(self):

        # Since we transformed mc -> latent ->  data / to invert the transformation we should use the inverse of the data 
        self.samples[:,2] = self.transform_data_var_2.invert_transform( self.samples[:,2], processed =True )
        self.samples[:,3] = self.transform_data_var_3.invert_transform( self.samples[:,3], processed = True )

        self.data_inputs[:,2] = self.transform_data_var_2.invert_transform( self.data_inputs[:,2] )
        self.data_inputs[:,3] = self.transform_data_var_3.invert_transform( self.data_inputs[:,3] )  

        # Since here we want both simulation and data at unstranformed for the final validation, we transform each other
        self.test_inputs[:,2][self.test_conditions[:,3] == 0]  = self.transform_sim_var_2.invert_transform( self.test_inputs[:,2][self.test_conditions[:,3] == 0] , processed =True )
        self.test_inputs[:,3][self.test_conditions[:,3] == 0]  = self.transform_sim_var_3.invert_transform( self.test_inputs[:,3][self.test_conditions[:,3] == 0] , processed =True ) 

        self.test_inputs[:,2][self.test_conditions[:,3] == 1] = self.transform_data_var_2.invert_transform( self.test_inputs[:,2][self.test_conditions[:,3] == 1], processed =True )
        self.test_inputs[:,3][self.test_conditions[:,3] == 1] = self.transform_data_var_3.invert_transform( self.test_inputs[:,3][self.test_conditions[:,3] == 1], processed =True ) 

    # Define and perform the training of the normalizing flow model
    def perform_training(self):
        
        # Defining the flow and changing the dtype to match the training arrays!
        if( self.Is_Autoregressive ):
            flow = zuko.flows.NSF( self.training_inputs.size()[1] , self.training_conditions.size()[1] , transforms=self.n_transforms, bins=self.n_splines_bins ,hidden_features=[self.aux_nodes] * self.aux_layers)
        else:
            flow = zuko.flows.NSF( self.training_inputs.size()[1] , self.training_conditions.size()[1] , transforms=self.n_transforms, bins=self.n_splines_bins ,hidden_features=[self.aux_nodes] * self.aux_layers, passes = 2)

        flow = flow.to( self.device )
        flow = flow.type( self.training_inputs.dtype )

        optimizer = torch.optim.Adam(flow.parameters(), self.initial_lr)
        early_stopper = EarlyStopper(patience = 15, min_delta=0.000)
        
        optimizer_plateu = torch.optim.Adam(flow.parameters(), self.initial_lr)
        scheduler_plateu = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_plateu, 'min', patience = 10)

        self.traning_loss_track = []
        self.validation_loss_track = []
        self.epoch_lr = []

        dataset = CustomDataset(self.training_inputs, self.training_conditions, self.training_weights)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if(self.cossine_annealing_scheduler):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

        print('\nBeginning of the flow training!\n')

        for epoch in range(999):
            
            for inputs, conditions, weights in dataloader:

                optimizer.zero_grad()
 
                loss = -flow( conditions ).log_prob( inputs )*weights 
                loss = loss.mean()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)

                optimizer.step()

                if( self.cossine_annealing_scheduler ):
                    scheduler.step()
            
            # End of the epoch! - calculating the validation loss and saving nescessary information!
            torch.save(flow.state_dict(), self.dump_folder + "/saved_states/epoch_" + str(epoch) + ".pth")
            with torch.no_grad():
                
                # Calculating the training loss with more events
                events = torch.randint(low=0, high= self.training_inputs.size()[0], size= (200000,))
                training_loss = -flow( self.training_conditions[events]).log_prob( self.training_inputs[events] )*self.training_weights[events]*1e6  
                training_loss = training_loss.mean()

                # Now the validation loss
                events = torch.randint(low=0, high= self.validation_inputs.size()[0], size= (200000,))
                validation_loss = -flow( self.validation_conditions[events]).log_prob( self.validation_inputs[events] )*self.validation_weights[events]*1e6  
                validation_loss = validation_loss.mean()

                if( self.cossine_annealing_scheduler ):
                    
                    # I am using cossine annealing and reduce on plateau to achieve a better convergence 
                    scheduler_plateu.step( validation_loss )
                    
                    for param_group in optimizer.param_groups:
                            param_group['lr'] = float(scheduler_plateu.optimizer.param_groups[0]['lr'])
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))
                    
                else:
                    scheduler.step( validation_loss )

                self.traning_loss_track.append(    float( training_loss )  )
                self.validation_loss_track.append( float( validation_loss ) )
                self.epoch_lr.append( scheduler.optimizer.param_groups[0]['lr'] )

                # Printing und updating the loss plot at each iteration
                print(f"\033[1;34mEpoch: {epoch}\033[0m, \033[1;32mTraining Loss: {training_loss:.3f}\033[0m, \033[1;31mValidation Loss: {validation_loss:.3f}\033[0m")
                self.plot_loss() 
                
                if( early_stopper.early_stop( float( validation_loss ) ) or epoch > self.max_epoch_number ):
                    print( 'Epoch with lowest loss: ', np.min(  np.array( self.validation_loss_track )  ) , ' at epoch: ', np.argmin( np.array( self.validation_loss_track ) ) )
                    
                    # We choose the epoch with the best validation loss for the performance assement
                    flow.load_state_dict(torch.load(self.dump_folder + '/saved_states/epoch_'+str(np.argmin( np.array( self.validation_loss_track ) )) +'.pth'))
                    torch.save(flow.state_dict(), self.dump_folder + "/best_model_.pth")
        
                    break

        self.flow = flow
        self.plot_loss()

    # Evaluates the performance of the flow in the test dataset after the training is finished
    def evaluate_flow( self ):

        # Lets select only MC events inside the test dataset
        self.MaskOnlytestMC       = self.test_conditions[:,self.test_conditions.size()[1] -1 ] == 0
        
        with torch.no_grad():

            trans      = self.flow(self.test_conditions[self.MaskOnlytestMC].to(self.device)).transform
            sim_latent = trans( self.test_inputs[self.MaskOnlytestMC].to(self.device) )

            self.test_conditions_ = torch.tensor(np.concatenate( [ self.test_conditions[self.MaskOnlytestMC][:,:-1].cpu() , np.ones_like( self.test_conditions[self.MaskOnlytestMC][:,0].cpu() ).reshape(-1,1) ], axis =1 )).to( self.device )

            trans2 = self.flow(self.test_conditions_ ).transform
            self.samples = trans2.inv( sim_latent)
        
            # Inverting the standartization transformation!
            self.invert_standartization()
            self.discontinous_variables_inverse_transformation()

            # Now I am brining the tensors back to cpu for plotting porpuses
            self.simulation_inputs = self.test_inputs[self.MaskOnlytestMC] #self.simulation_inputs.to('cpu')
            self.data_inputs       = self.data_inputs.to('cpu')
            self.samples           = self.samples.to('cpu') 
            self.test_weights      = self.test_weights[self.MaskOnlytestMC].to('cpu')  

            # saving the tensors for further BDT studies
            self.plot_data(evaluation = True)

            self.data_forid       = torch.cat( [ self.test_inputs[~self.MaskOnlytestMC].cpu() , self.test_conditions[~self.MaskOnlytestMC][:,:3].cpu() ] , axis = 1  )
            self.samples_forid    = torch.cat( [ self.samples.cpu()     , self.test_conditions[self.MaskOnlytestMC][:,:3].cpu() ] , axis = 1  )
            self.simulation_forid = torch.cat( [ self.test_inputs[self.MaskOnlytestMC].cpu() , self.test_conditions[self.MaskOnlytestMC][:,:3].cpu() ] , axis = 1  )

            #lets save these tensors
            torch.save( self.data_forid         , self.dump_folder + 'data.pt' )
            torch.save( self.samples_forid      , self.dump_folder + 'flow.pt' )
            torch.save( self.simulation_forid   , self.dump_folder + 'simulation.pt')    
            torch.save( self.test_weights       , self.dump_folder + 'weights.pt')         

    # Down here, only plotting stuff!
    def plot_data(self, evaluation = False, rw = False, transformation = False, validation = False):

        variables_to_plot = [ self.conditions_list, self.inputs]    
        if( evaluation ): 

            latex_inputs = [r'$v^\mathrm{A}_1$',r'$v^\mathrm{A}_2$',r'$v^\mathrm{B}_1$',r'$v^\mathrm{B}_2$']
            for i in range( self.samples.size()[1] ):

                    mean = np.mean(  np.array(self.data_inputs[:,i]))
                    std  = np.std(   np.array(self.data_inputs[:,i]))              
                    
                    if( 'B' in str(self.inputs[i]) and '1' in str(self.inputs[i])):
                        data_hist            = hist.new.Reg(25, 0.00, 6.0, underflow=True,overflow=True).Weight()  
                        mc_hist              = hist.new.Reg(25, 0.00, 6.0, underflow=True,overflow=True).Weight()  
                        mc_corr_hist         = hist.new.Reg(25, 0.00, 6.0, underflow=True,overflow=True).Weight()          
                    elif( '2' in str(self.inputs[i]) and 'B' in str(self.inputs[i])):
                        data_hist            = hist.new.Reg(25, 0.00, 7.0, underflow=True,overflow=True).Weight()  
                        mc_hist              = hist.new.Reg(25, 0.00, 7.0, underflow=True,overflow=True).Weight()  
                        mc_corr_hist         = hist.new.Reg(25, 0.00, 7.0, underflow=True,overflow=True).Weight()  
                    else:
                        data_hist            = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight() 
                        mc_hist              = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight() 
                        mc_corr_hist         = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight() 
                
                    self.test_weights = self.test_weights/torch.sum(self.test_weights)

                    #data_hist.fill( np.array(self.data_inputs[:,i] ) )
                    data_hist.fill(  np.array(self.test_inputs[~self.MaskOnlytestMC][:,i].cpu() )  )
                    mc_hist.fill(    np.array(self.test_inputs[self.MaskOnlytestMC][:,i].cpu() )  , weight = len(self.test_inputs[~self.MaskOnlytestMC])*self.test_weights.cpu() )
                    mc_corr_hist.fill( np.array(self.samples[:,i].cpu())       , weight = len(self.test_inputs[~self.MaskOnlytestMC])*self.test_weights.cpu() )

                    try:
                        os.makedirs( self.dump_folder + 'flow_results/', exist_ok=True )
                    except:
                        print( '\nIt was not possible to create the evaluation folder or the folder already exists. - exiting' )

                    plotter.plott( data_hist , mc_hist, mc_corr_hist ,self.dump_folder + 'flow_results/' + str(self.inputs[i]) +".png", xlabel = str(latex_inputs[i])  )

            # Lets also plot the reweigthed conditions for validation ...
            for key in self.conditions_list:
                
                mean = np.mean(  np.array(self.data[key]))
                std  = np.std(   np.array(self.data[key]))              

                if( 'oise' in str(key) ):
                    data_hist            = hist.new.Reg(80, 0.0, 3.0).Weight() 
                    mc_hist              = hist.new.Reg(80, 0.0, 3.0).Weight() 
                    mc_rw_hist           = hist.new.Reg(80, 0.0, 3.0).Weight() 
                else:
                    data_hist            = hist.new.Reg(40, 0.0, mean + 3.0*std).Weight() 
                    mc_hist              = hist.new.Reg(40, 0.0, mean + 3.0*std).Weight() 
                    mc_rw_hist           = hist.new.Reg(40, 0.0, mean + 3.0*std).Weight() 

                self.test_weights = self.test_weights/torch.sum(self.test_weights)

                data_hist.fill( np.array(self.data[key] ) )
                mc_hist.fill( np.array(self.simulation[key]) )
                mc_rw_hist.fill( np.array(self.simulation[key]) ,weight = len(self.data[key])*self.simulation_weights )

                plotter.plott( data_hist , mc_hist, mc_rw_hist ,self.dump_folder + 'flow_results/' + str(key) +".png", xlabel = str(key)  )

    def plot_transformation_before_and_after(self):
        
        # Makes the plots of the input distributions before and after the pre-processing transformations
        mc_before = self.simulation_inputs_before_transformations
        data_before = self.data_inputs_before_transformation

        MaskOnlyvalidationMC   = self.validation_conditions[:,self.validation_conditions.size()[1] -1 ] == 0
        self.mc_val_conditions = self.validation_conditions[MaskOnlyvalidationMC]
        mc_after    = self.validation_inputs[MaskOnlyvalidationMC]

        MaskOnlyvalidationData  = self.validation_conditions[:,self.validation_conditions.size()[1] -1 ] == 1
        self.data_val_conditions  = self.validation_conditions[MaskOnlyvalidationData]
        data_after       = self.validation_inputs[MaskOnlyvalidationData]

        # Now loop trough it!
        for i in range( mc_after.size()[1]  ):
            plotter.plot_transformations(data_before[:,i].detach().numpy(), data_after[:,i].detach().numpy(), mc_before[:,i].detach().numpy(), mc_after[:,i].detach().numpy(), "test", str(self.inputs[i]))

    # plot the loss of the training process!
    def plot_loss(self):

        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(self.traning_loss_track, color=color, marker='o', label='Training Loss')
        ax1.plot(self.validation_loss_track, color='tab:orange', marker='x', label='Validation Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(ax1.get_ylim()[0], 1.05*ax1.get_ylim()[1])
        #ax1.set_yscale('log')

        ax1.legend()

        # Create a second y-axis for 'initial' epoch learning rate
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('learning rate', color=color)
        ax2.plot(self.epoch_lr, color=color, marker='s', label='lr')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc = 'upper left')
        ax2.set_ylim(ax2.get_ylim()[0], 1.05*ax2.get_ylim()[1])

        # Title and show the plot
        plt.title('Loss Curve and learning rate')

        plt.tight_layout()
        plt.savefig(self.dump_folder+'loss_plot.png') 

