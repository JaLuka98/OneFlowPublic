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

    # Now, lets set up the flow model
    device = torch.device('cpu')

    # Lets read the .json containing the flow information
    with open('flow_configuration.json', 'r') as f:
        config = json.load(f)

    # Seting up the flow model
    flow = zuko.flows.NSF( 2 , 3 , transforms=config['n_transforms'], bins=config['n_spline_bins'] ,hidden_features=[config['aux_nodes']] * config['aux_layers'])
    flow.load_state_dict(torch.load('./model_three.pth'))

    #turning off graident calculation for flow evaluation
    with torch.no_grad():
        
        # Loading the test dataset 
        Treeway_inputs = torch.load('Treeway_inputs.pt') 
        Treeway_conditions = torch.load('Treeway_conditions.pt')

        mask_checkerboard = (Treeway_conditions == torch.tensor([0, 1, 0])).all(dim=1)#Treeway_conditions[:,:,:] == [0,1,0]
        Checkerboard = Treeway_inputs[mask_checkerboard]
        Checkerboard_conditions = Treeway_conditions[mask_checkerboard]

        # Checker to make moons!
        trans = flow(Checkerboard_conditions).transform
        sim_latent = trans( Checkerboard )

        simulation_conditions = torch.Tensor(np.concatenate([np.zeros_like(Checkerboard_conditions[:,0]).reshape(-1,1)   , np.zeros_like(Checkerboard_conditions[:,0]).reshape(-1,1)   , np.ones_like(Checkerboard_conditions[:,0]).reshape(-1,1)], axis = 1)).to(device)

        # From base distribution to four-circles space
        trans2 = flow(simulation_conditions).transform
        Checker_to_makemoons = trans2.inv( sim_latent)

        # Now checker to four-circles
        trans = flow(Checkerboard_conditions).transform
        sim_latent = trans( Checkerboard )

        simulation_conditions = torch.Tensor( np.concatenate([np.ones_like(Checkerboard_conditions[:,0]).reshape(-1,1)  , np.zeros_like(Checkerboard_conditions[:,0]).reshape(-1,1) , np.zeros_like(Checkerboard_conditions[:,0]).reshape(-1,1)], axis = 1) ).to(device)

        # From base distribution to four-circles space
        trans2 = flow(simulation_conditions).transform
        Checker_to_four_circles = trans2.inv( sim_latent)

        #############################################
        ## Make moons to others!
        
        mask_Make_moons = (Treeway_conditions == torch.tensor([0, 0, 1])).all(dim=1)
        Make_moons = Treeway_inputs[mask_Make_moons]
        Make_moons_conditions = Treeway_conditions[mask_Make_moons]

        # Make moons to four circles!
        trans = flow(Make_moons_conditions).transform
        sim_latent = trans( Make_moons )

        simulation_conditions = torch.Tensor( np.concatenate([np.ones_like(Make_moons_conditions[:,0]).reshape(-1,1)  , np.zeros_like(Make_moons_conditions[:,0]).reshape(-1,1) , np.zeros_like(Make_moons_conditions[:,0]).reshape(-1,1)], axis = 1) ).to(device)

        trans2 = flow(simulation_conditions).transform
        Makemoons_to_four_circles = trans2.inv( sim_latent)

        # Now to checkerboard
        trans = flow(Make_moons_conditions).transform
        sim_latent = trans( Make_moons )

        simulation_conditions = torch.Tensor( np.concatenate([np.zeros_like(Make_moons_conditions[:,0]).reshape(-1,1) , np.ones_like(Make_moons_conditions[:,0]).reshape(-1,1)  , np.zeros_like(Make_moons_conditions[:,0]).reshape(-1,1)], axis = 1) ).to(device)

        trans2 = flow(simulation_conditions).transform
        Makemoons_to_checkerboard = trans2.inv( sim_latent)

        ####### End of Make moons!
        
        ###################################################
        #### Now the same with the four circles set!
        ###################################################

        mask_Four_circles = (Treeway_conditions == torch.tensor([1, 0, 0])).all(dim=1)
        Four_circles = Treeway_inputs[mask_Four_circles]
        Four_circles_conditions = Treeway_conditions[mask_Four_circles]
        
        # Now checker to Checker!
        trans = flow(Four_circles_conditions).transform
        sim_latent = trans( Four_circles )

        simulation_conditions = torch.Tensor(np.concatenate([np.zeros_like(Four_circles_conditions[:,0]).reshape(-1,1) , np.ones_like(Four_circles_conditions[:,0]).reshape(-1,1)  , np.zeros_like(Four_circles_conditions[:,0]).reshape(-1,1)], axis = 1)).to(device)

        trans2 = flow(simulation_conditions).transform
        Four_circles_to_checker = trans2.inv( sim_latent)

        ## Now to make moons!
        trans = flow(Four_circles_conditions).transform
        sim_latent = trans( Four_circles )

        simulation_conditions = torch.Tensor(np.concatenate([np.zeros_like(Four_circles_conditions[:,0]).reshape(-1,1)   , np.zeros_like(Four_circles_conditions[:,0]).reshape(-1,1)   , np.ones_like(Four_circles_conditions[:,0]).reshape(-1,1)], axis = 1)).to(device)

        trans2 = flow(simulation_conditions).transform
        Four_circles_make_moons = trans2.inv( sim_latent)

    ######################################
    ### Enhanced plot!
    #####################################

    # Now, plotting the results!
    fig, axs = plt.subplots(3, 3, figsize=(36, 30), gridspec_kw={'wspace': 0.30,'hspace': 0.15})

    axs[0,0].hist2d( np.array(Checkerboard[:,0].detach().cpu())               , np.array(Checkerboard[:,1].detach().cpu())               , range = [[-4.0,4.0],[-4.0,4.0]]  ,bins=(150, 150)  , density = True , label='input')
    axs[0,1].hist2d( np.array(Checker_to_makemoons[:,0].detach().cpu())       , np.array(Checker_to_makemoons[:,1].detach().cpu() )      , range = [[-1.3,2.25],[-0.8,1.3]]  ,bins=(150, 150) , density = True , label='Data Points')
    axs[0,2].hist2d( np.array(Checker_to_four_circles[:,0].detach().cpu())    , np.array(Checker_to_four_circles[:,1].detach().cpu() )   , range = [[-3.5,3.5],[-3.5,3.5]]  ,bins=(150, 150)  , density = True , label='Data Points')

    axs[1,0].hist2d( np.array(Make_moons[:,0].detach().cpu())                 , np.array(Make_moons[:,1].detach().cpu())                  , range = [[-1.3,2.25],[-0.8,1.3]]  ,bins=(150, 150) , density = True , label='input')
    axs[1,1].hist2d( np.array(Makemoons_to_checkerboard[:,0].detach().cpu())  , np.array(Makemoons_to_checkerboard[:,1].detach().cpu() )  , range = [[-4.0,4.0],[-4.0,4.0]]  ,bins=(150, 150)  , density = True , label='Data Points')
    axs[1,2].hist2d( np.array(Makemoons_to_four_circles[:,0].detach().cpu())  , np.array(Makemoons_to_four_circles[:,1].detach().cpu() )  , range = [[-3.5,3.5],[-3.5,3.5]]  ,bins=(150, 150)  , density = True , label='Data Points')

    axs[2,0].hist2d( np.array(Four_circles[:,0].detach().cpu())               , np.array(Four_circles[:,1].detach().cpu())               , range = [[-3.5,3.5],[-3.5,3.5]]  ,bins=(150, 150)  , density = True , label='input')
    axs[2,1].hist2d( np.array(Four_circles_to_checker[:,0].detach().cpu())    , np.array(Four_circles_to_checker[:,1].detach().cpu() )   , range = [[-4.0,4.0],[-4.0,4.0]]  ,bins=(150, 150)  , density = True , label='Data Points')
    axs[2,2].hist2d( np.array(Four_circles_make_moons[:,0].detach().cpu())    , np.array(Four_circles_make_moons[:,1].detach().cpu() )   , range = [[-1.3,2.25],[-0.8,1.3]]  ,bins=(150, 150) , density = True , label='Data Points')

    # Remove ticks from all subplots
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    axs[0, 0].set_title('Original distributions',fontsize=70, fontweight='bold', pad=40)
    axs[0, 1].set_title('Flow morphed',fontsize=70, fontweight='bold', pad=40)
    axs[0, 2].set_title('Flow morphed',fontsize=70, fontweight='bold', pad=40)
               
    # Add spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.05, wspace=0.30, hspace=0.15)

    plt.savefig( 'three_datasets_enhanced.png' )
    plt.savefig( 'three_datasets_enhanced.pdf' )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Apply a trained NSF flows to HiggsDNA output")
    parser.add_argument('-yamlpath'  , '--yamlpath'  , type=str, help= "Path to the yaml file that contaings the yaml with the configurations")
    parser.add_argument('-dataset'   , '--dataset'   , type=str, help= "Fourcircles or Makemoons")
    parser.add_argument('-filespath' , '--filespath' , type=str, help= "Path to the toy data and simulation files")
    parser.add_argument('-outpath'   , '--outpath'   , type=str, help= "Path to dumop the results")
    args = parser.parse_args()

    main()