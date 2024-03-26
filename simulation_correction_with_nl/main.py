# Imports
import argparse
import yaml
from yaml import Loader

# Importing other scripts
import network_utils

# Main script to perform the corrections of the physics inspired dataset
def main():

    print("\n\033[92m🚀 Welcome to 'One Flow to Correct Them All' code! 🌌\033[0m\n\n")

    # Loop to read over network condigurations from the yaml file: 
    stream = open(args.yamlpath , 'r')
    dictionary = yaml.load(stream,Loader)

    for key in dictionary:

        # Configuration - name of the folder that will be created, and where the network states and plots will be saved
        configuration = args.outpath + str(key)

        #network configurations
        n_transforms   = dictionary[key]["n_transforms"]    # number of transformation
        aux_nodes      = dictionary[key]["aux_nodes"]       # number of nodes in the auxiliary network
        aux_layers     = dictionary[key]["aux_layers"]      # number of auxiliary layers in each flow transformation
        n_splines_bins = dictionary[key]["n_splines_bins"]  # Number of rationale quadratic spline flows bins

        # Now defining the parameters of the discontinious variables transformation - all the discontinious variables start at 1
        shift_discontinious_var1 = dictionary[key]["max_x_triang"]   # shift is the value until when the values of the triangular function will be sampled
        shift_discontinious_var2 = dictionary[key]["max_x_triang"]   # As the discontinious variables tails start from diferent values, the shift value can also be changed

        # Some general training parameters
        max_epoch_number = 999
        initial_lr       = dictionary[key]["initial_lr"]
        batch_size       = dictionary[key]["batch_size"]

        # Autoregressive or coupling blocks
        Is_Autoregressive = dictionary[key]["autoregressive"]

        # Calling the main class that performs from data reading to flow training and evaluation
        reader = network_utils.Toy_data_framework( args.filespath , configuration, n_transforms, n_splines_bins, aux_nodes, aux_layers, shift_discontinious_var1, shift_discontinious_var2, max_epoch_number, initial_lr, batch_size, Is_Autoregressive)
        reader.plot_data()
        reader.treat_data()
        reader.perform_training()
        reader.evaluate_flow()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Apply a trained NSF flows to HiggsDNA output")
    parser.add_argument('-yamlpath'  , '--yamlpath'  , type=str, help= "Path to the yaml file that contaings the yaml with the configurations")
    parser.add_argument('-filespath' , '--filespath' , type=str, help= "Path to the toy data and simulation files")
    parser.add_argument('-outpath'   , '--outpath'   , type=str, help= "Path to dump the results")
    args = parser.parse_args()

    main()