# importing some
import numpy as np
import matplotlib.pyplot as plt
import os
import xgboost as xgb
import pandas as pd
from matplotlib import pyplot
import argparse
import torch
from sklearn.model_selection import train_test_split

# additional scripts
import BDT_plot_utils as plot_utils

# Script to perform further multivariate check on the performance of the flow
def main():

    # Path where the plots will be stored
    path_to_plots = args.plotspath

    # Mkdir the path to store the plots
    os.makedirs(path_to_plots, exist_ok=True)

    # Loading the test data tensors for further processing - they are pytorch tensors
    data_forid    = torch.load( args.datafilespath )
    samples_forid = torch.load( args.flowfilespath )
    mc_forid      = torch.load( args.flowfilespath[:-7] + 'simulation.pt' )
    weights_forid = torch.load(args.weightspath)
    data_weights  = torch.tensor( torch.ones( len(data_forid) ) )       
    ######################################################################
    
    input_names = [r'$v^\mathrm{A}_1$',r'$v^\mathrm{A}_2$',r'$v^\mathrm{B}_1$',r'$v^\mathrm{B}_2$',r'$p_{\mathrm{T}}$',r'$\eta$',r'$N$']
    
    # Normalizing the simulation weights to data
    sim_weights  = torch.sum( data_weights ) * (weights_forid/torch.sum( weights_forid ))

    # first, lets plot some correlations matrices based on the flows results
    var_names  = [r'$v^\mathrm{A}_1$',r'$v^\mathrm{A}_2$',r'$v^\mathrm{B}_1$',r'$v^\mathrm{B}_2$',r'$p_{\mathrm{T}}$',r'$\eta$',r'$N$']
    
    plot_utils.plot_correlation_matrices(data_forid,mc_forid,samples_forid, weights_forid , var_names, path_to_plots )
    plot_utils.size_corrections(mc_forid.detach().numpy(), samples_forid.detach().numpy(), data_forid.detach().numpy() , weights_forid.detach().numpy(), var_names, path_to_plots)

    # building the tensors
    data_forid_label    = np.ones( len(data_forid) )
    samples_forid_label = 0*np.ones( len(samples_forid) )

    train_data    = np.concatenate( [ data_forid             , samples_forid ] , axis = 0 )
    train_labels  = np.concatenate( [ data_forid_label       , samples_forid_label ], axis = 0 )
    train_weights = np.concatenate( [ data_weights           , sim_weights ], axis = 0  )

    #now split the data into training and test sets, and also shuffle everything
    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(train_data, train_labels, train_weights, test_size=0.5, shuffle=True)

    # Creating the inputs that will be feed to the BDT training and evaluation
    train = xgb.DMatrix(data= x_train ,label= y_train, weight = w_train,
                    missing=-999.0, feature_names = input_names )

    test = xgb.DMatrix(data= x_test ,label = y_test,  weight = w_test , 
                    missing=-999.0, feature_names = input_names  )

    # Defining the parameters for the classifier and training
    param = {}

    # Booster parameters
    param['eta']              = 0.1   # learning rate
    param['max_depth']        = 10    # maximum depth of a tree
    param['subsample']        = 0.5   # fraction of events to train tree on
    param['colsample_bytree'] = 1.0   # fraction of features to train tree on
    # Any more max depth than this and it overfits hard!

    # Learning task parameters
    param['objective']   = 'binary:logistic' # objective function
    param['eval_metric'] = 'error'           # evaluation metric for cross validation
    param = list(param.items()) + [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]

    num_trees = 999 

    # Object to keep track of the losses
    evals_result = {}  
    evals = [(train, "train"), (test, "validation")]

    # perform the training. 
    print( '\nBeggining of the Flow vs Data BDT training\n' )
    booster = xgb.train(param,train,num_boost_round=num_trees, evals = evals, evals_result=evals_result, verbose_eval=10, early_stopping_rounds = 30)

    # Evaluating the BDT
    predictions = booster.predict(test)
    predictions_train = booster.predict(train)

    # Printing the importance of every variable
    xgb.plot_importance(booster,grid=False)
    plt.savefig( path_to_plots + 'BDT_importance.png' ) #saving the plot

    # Plotting the background and signal BDT scores 
    plot_utils.plot_BDT_output( predictions      , test,  path_to_plots  )
    plot_utils.plot_BDT_output( predictions_train, train, path_to_plots, trainset = True  )

    #lets also plot the loss curve for the model
    plot_utils.plot_model_loss( evals_result , path_to_plots )

    #saving the model in .json format
    booster.save_model("bdt_data_vs_flow.json")
    
    # Testing profiles for train and test samples
    plot_utils.pt_profile_again( predictions, x_test, y_test, path_to_plots, w_test )

    ##########################################################################################################
    # For the ROC curve comparing the MC and MC corrected ROC cruves, we need to train a model also In pure MC
    ##########################################################################################################

    train_data    = np.concatenate( [ data_forid             , mc_forid ] , axis = 0 )
    train_labels  = np.concatenate( [ data_forid_label       , samples_forid_label ], axis = 0 )
    train_weights = np.concatenate( [ data_weights           , sim_weights ], axis = 0  )

    #now split the data into training and test sets, and also shuffle everything
    x_train, x_test, y_train, y_test, w_train_unc, w_test_unc = train_test_split(train_data, train_labels, train_weights, test_size=0.5, shuffle=True)

    # Creating the inputs that will be feed to the BDT training and evaluation
    train_unc = xgb.DMatrix(data= x_train ,label= y_train, weight = w_train_unc,
                    missing=-999.0, feature_names = input_names )

    test_unc = xgb.DMatrix(data= x_test ,label = y_test,  weight = w_test_unc , 
                    missing=-999.0, feature_names = input_names  )

    # Defining the parameters for the classifier and training
    param = {}

    # Booster parameters
    param['eta']              = 0.1   # learning rate
    param['max_depth']        = 10    # maximum depth of a tree
    param['subsample']        = 0.5   # fraction of events to train tree on
    param['colsample_bytree'] = 1.0   # fraction of features to train tree on

    # Learning task parameters
    param['objective']   = 'binary:logistic' # objective function
    param['eval_metric'] = 'error'           # evaluation metric for cross validation
    param = list(param.items()) + [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]

    num_trees = 999  # number of trees to make

    evals_result = {}  # object to keep track of the losses
    evals = [(train_unc, "train"), (test_unc, "validation")]

    # perform the training of the BDT
    print( '\nBeggining of the Nominal MC vs Data BDT training\n' )
    booster_unc = xgb.train(param, train_unc, num_boost_round=num_trees, evals = evals, evals_result=evals_result, verbose_eval=10, early_stopping_rounds = 30)

    plot_utils.ROC_trainig_and_validation_curves(train, train.get_label(), test, test.get_label(), w_train, w_test, booster, train_unc, train_unc.get_label(), test_unc, test_unc.get_label(), w_train_unc, w_test_unc, booster_unc, path_to_plots  )
               

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a BDT to classify toy data events into MC or data.")
    parser.add_argument('-datafilespath'   , '--datafilespath'  , type=str, help="Path to the toy data files.", required = True)
    parser.add_argument('-flowfilespath'   , '--flowfilespath'  ,type=str, help="Path to the toy MC files.", required = True)
    parser.add_argument('-plotspath'       , '--plotspath'      ,type=str, help="Path where the plots will be stored.", required = True)
    parser.add_argument('-weight'          , '--weightspath'    ,type=str, help="Path to the weights tensor for the reweithed samples.", required = True)
    args = parser.parse_args()

    main() 