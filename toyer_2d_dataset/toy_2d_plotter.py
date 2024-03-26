# standard imports
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

import os
import torch
import zuko
import json

from sklearn.datasets import make_moons
import datasets_generation as data_gen

# Script to load and plot the results of the toy_2d_dataset script
def main():

    four_circles_dataset = True if (args.dataset == "Fourcircles") else False

    # Now, lets set up the flow model
    device = torch.device('cpu')

    # Lets read the .json containing the flow information
    with open('flow_configuration.json', 'r') as f:
        config = json.load(f)

    # Seting up the flow model
    flow = zuko.flows.NSF( 2 , 1 , transforms=config['n_transforms'], bins=config['n_spline_bins'] ,hidden_features=[config['aux_nodes']] * config['aux_layers'])
    
    if four_circles_dataset:
        flow.load_state_dict(torch.load('./model_fourcircles.pth'))
    else:
        flow.load_state_dict(torch.load('./model_make_moons.pth'))

    # Sampling the new events for performance evaluation
    n_test_points = 200000

    #turning off graident calculation for flow evaluation
    with torch.no_grad():

        # Generating and preparint it for the flows
        if four_circles_dataset:
            """
            # Generating more data for the evaluation phase
            simulation = data_gen.CheckerboardDataset( n_test_points )
            simulation = np.concatenate( [np.array(simulation[:,0]).reshape(-1,1),np.array(simulation[:,1]).reshape(-1,1)], axis = 1)
            simulation_conditions = (np.ones_like(simulation[:,0])*0).reshape(-1,1)
            
            simulation, simulation_conditions = torch.Tensor( simulation ).to(device) , torch.Tensor( simulation_conditions ).to(device)
            """
            # Loading the test dataset 
            simulation_inputs = torch.load('Four_circles_inputs.pt') 
            simulation_conditions = torch.load('Four_circles_conditions.pt')
            
            # Selecting only the make moon events
            mask = torch.squeeze(simulation_conditions == 0, axis =1)
            simulation = simulation_inputs[mask]
            simulation_conditions = simulation_conditions[mask]

        else:
            # Loading the test dataset 
            simulation_inputs = torch.load('Make_moons_inputs.pt') 
            simulation_conditions = torch.load('Make_moons_conditions.pt')
            
            # Selecting only the make moon events
            mask = torch.squeeze(simulation_conditions == 0, axis =1)
            simulation = simulation_inputs[mask]
            simulation_conditions = simulation_conditions[mask]

        # Transforming the CheckerboardDataset into the four circles or Make Moons!
        trans = flow(simulation_conditions).transform
        sim_latent = trans( simulation )

        # Inverting the boolean
        simulation_conditions = torch.Tensor((np.ones_like(simulation[:,0].cpu())).reshape(-1,1)).to(device)

        # From base distribution to four-circles space
        trans2 = flow(simulation_conditions).transform
        samples = trans2.inv( sim_latent)

    # Now we generate four circles and trasform it to Checkerboard Dataset distribution
    with torch.no_grad():

        # Generating and preparint it for the flows
        if four_circles_dataset:
            #data = data_gen.FourCircles( n_test_points )

            # Loading the test dataset
            data_inputs = torch.load('Four_circles_inputs.pt') 
            data_conditions = torch.load('Four_circles_conditions.pt')
            
            # Selecting only the make moon events
            mask = torch.squeeze(data_conditions == 1, axis =1)
            data = data_inputs[mask]
            data_conditions = data_conditions[mask]
        else:
            # Loading the test dataset
            data_inputs = torch.load('Make_moons_inputs.pt') 
            data_conditions = torch.load('Make_moons_conditions.pt')
            
            # Selecting only the make moon events
            mask = torch.squeeze(data_conditions == 1, axis =1)
            data = data_inputs[mask]
            data_conditions = data_conditions[mask]
            
            #data, labels = make_moons(n_test_points, noise=0.05)
        
        #data       = np.concatenate( [np.array(data[:,0]).reshape(-1,1), np.array(data[:,1]).reshape(-1,1)] , axis = 1)
        #data_conditions       = np.ones_like(data[:,0]).reshape(-1,1)

        #data, data_conditions = torch.Tensor( data ).to(device) , torch.Tensor( data_conditions ).to(device)

        # Transforming the four circles into the four Checkerboard Dataset!
        trans = flow(data_conditions).transform
        data_latent = trans( data )

        data_conditions = torch.Tensor((np.ones_like(data[:,0].cpu())*0).reshape(-1,1)).to(device)

        trans2 = flow(data_conditions).transform
        samples_2 = trans2.inv( data_latent)

    # Now, plotting the results!
    fig, axs = plt.subplots(2, 2, figsize=(36, 30), gridspec_kw={'wspace': 0.30,'hspace': 0.15})
    axs[0,0].hist2d( np.array(simulation[:,0].detach().cpu())       , np.array(simulation[:,1].detach().cpu())      , range = [[-4.0,4.0],[-4.0,4.0]]  ,bins=(150, 150), density = True , label='input')
    if four_circles_dataset:
        axs[1,0].hist2d( np.array(data[:,0].detach().cpu())             , np.array(data[:,1].detach().cpu() )           , range = [[-3.5,3.5],[-3.5,3.5]]  ,bins=(150, 150), density = True ,label='Data Points')
    else: 
        axs[1,0].hist2d( np.array(data[:,0].detach().cpu())             , np.array(data[:,1].detach().cpu() )           , range = [[-1.3,2.25],[-0.8,1.3]]  ,bins=(150, 150), density = True ,label='Data Points')

    axs[1,1].hist2d( np.array(samples_2[:,0].detach().cpu())        , np.array(samples_2[:,1].detach().cpu())       , range = [[-4.0,4.0],[-4.0,4.0]]  ,bins=(150, 150), density = True ,label='Simulation Points')
    if four_circles_dataset:
        axs[0,1].hist2d( np.array(samples[:,0].detach().cpu())          , np.array(samples[:,1].detach().cpu() )        , range = [[-3.5,3.5],[-3.5,3.5]]  ,bins=(150, 150), density = True ,label='Simulation Points')
    else: 
        axs[0,1].hist2d( np.array(samples[:,0].detach().cpu())          , np.array(samples[:,1].detach().cpu() )        , range = [[-1.3,2.25],[-0.8,1.3]]  ,bins=(150, 150), density = True ,label='Simulation Points')

    # Remove ticks from all subplots
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    axs[0, 0].set_title('Original distributions',fontsize=90, fontweight='bold', pad=40)
    axs[0, 1].set_title('Flow morphed',fontsize=90, fontweight='bold', pad=40)

    # Adding an arrow from close but outside of the axs[1,0] to close but outside of the axs[0,1]
    plt.annotate('', xy=(0.00, 1.7), xycoords='axes fraction', xytext=(-0.25, 1.7), textcoords='axes fraction', 
                    arrowprops=dict(arrowstyle="-|>", color='blue',  mutation_scale=75, linewidth=35), annotation_clip=False)

    plt.annotate('', xy=(0.00, 0.4), xycoords='axes fraction', xytext=(-0.25, 0.4), textcoords='axes fraction', 
                    arrowprops=dict(arrowstyle="-|>", color='blue',  mutation_scale=75, linewidth=35), annotation_clip=False)
                    
    # Add spacing between subplots
    # Adjust layout manually
    plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.05, wspace=0.30, hspace=0.15)
    #plt.tight_layout(pad=0)
    #plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    
    if four_circles_dataset:
        plt.savefig( 'Chessboard_fourcircles.png' )
    else:
        plt.savefig( 'Chessboard_makemoons.png' )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Apply a trained NSF flows to HiggsDNA output")
    parser.add_argument('-yamlpath'  , '--yamlpath'  , type=str, help= "Path to the yaml file that contaings the yaml with the configurations")
    parser.add_argument('-dataset'   , '--dataset'   , type=str, help= "Fourcircles or Makemoons")
    parser.add_argument('-filespath' , '--filespath' , type=str, help= "Path to the toy data and simulation files")
    parser.add_argument('-outpath'   , '--outpath'   , type=str, help= "Path to dumop the results")
    args = parser.parse_args()

    main()
