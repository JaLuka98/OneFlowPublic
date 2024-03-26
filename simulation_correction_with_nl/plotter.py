# Standart import
import scipy
import numpy as np
import matplotlib.pyplot as plt
import mplhep, hist
plt.style.use([mplhep.style.CMS])
import mplhep as hep
from matplotlib.ticker import FuncFormatter, MaxNLocator
import zuko
import torch
import argparse
import os
from matplotlib.ticker import FuncFormatter

plt.rc('font', family='serif', size=18)  # Set default font family and size
plt.rc('text', usetex=True)

#plt.rcParams['text.usetex'] = True

import warnings
warnings.filterwarnings("ignore")

def to_int(x, pos):
    # Certifique-se de que o valor é um inteiro e dentro de um intervalo razoável
    try:
        value = int(round(x))
    except ValueError:  # Em caso de falha, retorne uma string vazia
        return ''
    return f'{value}'

#plt.rcParams['text.usetex'] = True

#main plot function!
def plott(data_hist, mc_hist, mc_corr_hist ,output_filename, xlabel,region=None  ):

    """
    Generates a two-panel plot with histograms for data, Monte Carlo (MC) simulation, and reweighted MC simulation. 
    The top panel shows the normalized histograms for each dataset, and the bottom panel displays the ratio of data 
    to MC simulation and data to reweighted MC simulation.

    Parameters:
    - data_hist: Histogram object for the data. This object should be compatible with hep.histplot.
    - mc_hist: Histogram object for the MC simulation. This object should be compatible with hep.histplot.
    - mc_corr_hist: Histogram object for the reweighted MC simulation. This object should be compatible with hep.histplot.
    - output_filename: String specifying the filename (including path) where the plot will be saved.
    - xlabel: String specifying the label for the x-axis.
    - region: Optional; String specifying the region or selection criteria used for the dataset. This information is displayed as text on the plot. Default is None.
    - all_blue: Optional; Boolean flag to indicate whether to plot both MC and reweighted MC histograms in blue. If False, the MC histogram is plotted in blue, and the reweighted MC histogram is plotted in red. Default is False.

    The function creates a figure with two subplots arranged vertically. The top subplot displays the normalized histograms for the data (in black), MC simulation (in blue or red), and optionally the reweighted MC simulation (in red if all_blue is False). The bottom subplot shows the ratio of data to MC and data to reweighted MC, highlighting the agreement between the data and simulations.

    The histograms are normalized to unity, and the ratio plots include error bars representing the statistical uncertainties in the data. The function saves the generated plot to the specified output filename.

    Note: This function requires matplotlib and mplhep libraries for plotting and assumes the histograms are provided in a format compatible with hep.histplot.
    """

    fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [5,1, 1], 'wspace':0.12, 'hspace':0.12}, sharex=True)
    
    hep.histplot(
        mc_hist,
        label = r'Toy simulation',
        yerr=np.sqrt(mc_hist.variances()),
        density = True,
        color = "blue",
        linewidth=3,
        ax=ax[0],
        flow = 'sum'
    )

    hep.histplot(
        mc_corr_hist,
        label = r'Corrected toy simulation',
        yerr= np.sqrt(mc_corr_hist.variances()),
        density = True,
        color = "red",
        linewidth=3,
        ax=ax[0],
        flow = 'sum'
    )

    hep.histplot(
        data_hist,
        label = "Toy data",
        yerr=np.sqrt(data_hist.variances()),
        density = True,
        color="black",
        histtype='errorbar',
        markersize=13,
        elinewidth=3,
        ax=ax[0],
        flow = 'sum'
    )

    ax[0].set_xlabel('')
    #ax[0].margins(y=0.15)
    if('A' in xlabel ):
        ax[0].set_ylim(0, 1.35*ax[0].get_ylim()[1])
    else:
        ax[0].set_ylim(0, 1.05*ax[0].get_ylim()[1])
    ax[0].tick_params(labelsize=22)
    
    ax[0].margins(x=0)
    ax[1].margins(x=0)
    ax[2].margins(x=0)

    # line at 1
    ax[1].axhline(1, 0, 1, label=None, linestyle='--', color="black", linewidth=2)#, alpha=0.5)

    data_hist_numpy = data_hist.to_numpy()
    mc_hist_numpy   = mc_hist.to_numpy()
    mc_hist_rw_numpy   = mc_corr_hist.to_numpy()
    
    # Calculating the ratio betwenn nominal mc and data and the stat error of mc + data   
    values1, variances1 = data_hist.values() , data_hist.variances()
    values2, variances2 = mc_hist.values()   , mc_hist.variances()

    # Calculate the ratio 
    ratio = values1 / values2

    # MC nominal error
    error_ratio = np.sqrt((np.sqrt(variances1) / values1) ** 2 + (np.sqrt(variances2) / values2) ** 2) #np.sqrt(variances2) / values2 #np.sqrt(variances2) / values2

    hep.histplot(
        ratio,
        bins=data_hist_numpy[1],
        label=None,
        color="blue",
        histtype='errorbar',
        yerr=error_ratio,
        markersize=14,
        elinewidth=3,
        alpha=1,
        ax=ax[1]
    )

    hep.histplot(
        ratio,
        bins=data_hist_numpy[1],
        label=None,
        color="blue",
        histtype='errorbar',
        yerr=error_ratio,
        markersize=14,
        elinewidth=3,
        alpha=1,
        ax=ax[2]
    )

    # Calculating the ratio betwenn nominal mc and data and the stat error of mc + data
    values1, variances1 = data_hist.values()  , data_hist.variances() 
    values2, variances2 = mc_corr_hist.values() , mc_corr_hist.variances()

    # Calculate the ratio 
    ratio_rw = values1 / values2

    # MC nominal error
    error_ratio_rw = ratio_rw*np.sqrt((np.sqrt(variances1) / values1) ** 2 + (np.sqrt(variances2) / values2) ** 2) #np.sqrt(variances2) / values2

    hep.histplot(
        ratio_rw,
        bins=data_hist_numpy[1],
        label=None,
        color="red",
        histtype='errorbar',
        yerr=error_ratio_rw,
        markersize=14,
        elinewidth=3,
        alpha=1,
        ax=ax[1],
    )


    hep.histplot(
        ratio_rw,
        bins=data_hist_numpy[1],
        label=None,
        color="red",
        histtype='errorbar',
        yerr=error_ratio_rw,
        markersize=14,
        elinewidth=3,
        alpha=1,
        ax=ax[2],
    )


    # Function to format y-axis values to two decimals
    def format_y_axis(value, _):
        return '{:.2f}'.format(value)

    # Apply the formatter to ax[1]
    ax[1].yaxis.set_major_formatter(FuncFormatter(format_y_axis))

    ax[0].set_ylabel(r'\textbf{Normalized to unity}', fontsize=24)
    ax[1].set_ylabel(r'\textbf{Sim./data}', loc='center', family='Helvetica', fontsize=18)
    ax[2].set_ylabel(r'\textbf{Sim./data}', loc='center', family='Helvetica', fontsize=18)
    ax[2].set_xlabel( str(xlabel) , fontsize=26)

    ax[0].tick_params(labelsize=22)
    ax[2].tick_params(axis='x', labelsize=22)
    #ax[2].xtick_params(labelsize=22)
    #ax.set_ylim(0., 1.1*ax.get_ylim()[1])
    ax[1].set_ylim(0.79, 1.21)
    ax[2].set_ylim(0.945, 1.055)

    ax[2].axhline(1, 0, 1, label=None, linestyle='--', color="black", linewidth=2)

    ax[2].axhline(1.01, 0, 1.01, label=None, color="blue", linewidth=2)
    ax[2].axhline(1.02, 0, 1.02, label=None, linestyle='--', color="blue", linewidth=2)

    ax[2].axhline(0.99, 0, 0.99, label=None, color="blue", linewidth=2)
    ax[2].axhline(0.98, 0, 0.98, label=None, linestyle='--', color="blue", linewidth=2)

    ax[0].legend(
        loc="upper right",
        prop={'family': 'Helvetica', 'size': 18}  # Adjust the size as needed
    )
    
    #ax[1].set_xticklabels([])
    #ax[0].set_yticklabels([])

    #plt.subplots_adjust(pad=-5.0)
    #plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=-1.0, h_pad=0.5, w_pad=0.5)
    #plt.tight_layout()

    fig.savefig(output_filename, bbox_inches='tight')


def plot_transformations(data_before,data_after,mc_before,mc_after,path,name):

    # From numpy arrays to hist histograms =p
    mean_before = np.mean( np.array(data_before))
    std_before  = np.std(  np.array(data_before))

    mean_after = np.mean( np.array(data_after))
    std_after  = np.std(  np.array(data_after))

    # Creating the histograms
    if( 'B' in name ):
        data_hist_before    = hist.new.Reg(70, 0.0, mean_before + 3.0*std_before, overflow=True).Weight()
        mc_hist_before      = hist.new.Reg(70, 0.0, mean_before + 3.0*std_before, overflow=True).Weight()
    else:
        data_hist_before    = hist.new.Reg(70, mean_before - 3.0*std_before, mean_before + 3.0*std_before, overflow=True).Weight()
        mc_hist_before      = hist.new.Reg(70, mean_before - 3.0*std_before, mean_before + 3.0*std_before, overflow=True).Weight()

    data_hist_after      = hist.new.Reg(70, mean_after - 3.0*std_after, mean_after + 2.0*std_after, overflow=True).Weight()
    mc_hist_after        = hist.new.Reg(70, mean_after - 3.0*std_after, mean_after + 2.0*std_after, overflow=True).Weight()

    # Now filling the histograms
    data_hist_before.fill( np.array( data_before))
    mc_hist_before.fill(   np.array(mc_before))

    data_hist_after.fill( np.array( data_after))
    mc_hist_after.fill(   np.array(mc_after))

    # Now, plotting the results!
    fig, axs = plt.subplots(1, 2, figsize=(38, 14) , gridspec_kw={'wspace': 0.22,'hspace': 0.00})

    # Before plots
    hep.histplot(
            mc_hist_before,
            label = r'Simulation',
            yerr=False,
            density = True,
            color = "blue",
            linewidth=5,
            ax=axs[0],
            flow = 'sum'
        )

    hep.histplot(
        data_hist_before,
        label = "Data",
        yerr=True,
        density = True,
        color="black",
        histtype='errorbar',
        markersize=18,
        elinewidth=5,
        ax=axs[0],
        flow = 'sum'
    )

    # After the transformation
    hep.histplot(
            mc_hist_after,
            label = r'Simulation',
            yerr=False,
            density = True,
            color = "blue",
            linewidth=5,
            ax=axs[1],
            flow = 'sum'
        )

    hep.histplot(
        data_hist_after,
        label = "Data",
        yerr=True,
        density = True,
        color="black",
        histtype='errorbar',
        markersize=18,
        elinewidth=5,
        ax=axs[1],
        flow = 'sum'
    )

    axs[0].set_xlabel('')
    axs[0].margins(y=0.15)
    axs[0].set_ylim(0, 1.1*axs[0].get_ylim()[1])
    axs[0].tick_params(labelsize=35)

    axs[1].set_xlabel('')
    axs[1].margins(y=0.15)
    axs[1].set_ylim(0, 1.1*axs[1].get_ylim()[1])
    axs[1].tick_params(labelsize=35)
    
    axs[0].margins(x=0)
    axs[1].margins(x=0)

    axs[0].set_title(r'Before smoothing',fontsize=62, fontweight='bold', pad=40)
    axs[1].set_title(r'After smoothing',fontsize=62, fontweight='bold', pad=40)

    axs[0].set_ylabel("Normalized to unity", fontsize=40)
    axs[1].set_ylabel("Normalized to unity", fontsize=40)
    
    #plt.rcParams['text.usetex'] = True
    axs[0].set_xlabel( r'discontinious_var' , fontsize=48)
    axs[1].set_xlabel( r'discontinious_var' , fontsize=48)

    axs[0].legend(
        loc="upper right", fontsize=36
    )
    axs[1].legend(
        loc="upper right", fontsize=36
    )

    # Accessing the spines for each subplot and adjusting the thickness of the frame
    for ax in axs:
        ax.spines['top'].set_linewidth(4)    # Top frame
        ax.spines['right'].set_linewidth(4)  # Right frame
        ax.spines['bottom'].set_linewidth(4) # Bottom frame
        ax.spines['left'].set_linewidth(4)   # Left frame

    plt.tight_layout()

    try:
        os.makedirs( path,  exist_ok=True)
    except:
        print('\nIt was not possible to open the ./transformations/ folder')
        exit()

    plt.savefig( path + name+'_transform.pdf' )

# Plotting functions to plot the marginal distributions
def main():

    device = torch.device('cpu') 

    # Lets read the validation dataset
    data_forid    = torch.load( args.path + 'data.pt')
    samples_forid = torch.load( args.path + 'flow.pt')
    mc_forid      = torch.load( args.path + 'simulation.pt' )
    weights_forid = torch.load( args.path + 'weights.pt')
    data_weights  = torch.tensor( torch.ones( len(data_forid) ) ) 

    # Making the plots of the marginal distributions!
    latex_inputs = [r'$v^\mathrm{A}_1$',r'$v^\mathrm{A}_2$',r'$v^\mathrm{B}_1$',r'$v^\mathrm{B}_2$',r'$p_{T}$',r'$\eta$',r'$N$']
    inputs       = ["v1A","v2A","v1B","v2B","pT","eta","noise"]
    for i in range( samples_forid.size()[1] ):

        mean = np.mean(  np.array(data_forid[:,i]))
        std  = np.std(   np.array(data_forid[:,i]))

        if( 'B' in str(inputs[i]) and '1' in str(inputs[i])):
            data_hist            = hist.new.Reg(25, 0.00, 6.0, underflow=True,overflow=True).Weight()  
            mc_hist              = hist.new.Reg(25, 0.00, 6.0, underflow=True,overflow=True).Weight()  
            mc_corr_hist         = hist.new.Reg(25, 0.00, 6.0, underflow=True,overflow=True).Weight()          
        elif( '2' in str(inputs[i]) and 'B' in str(inputs[i])):
            data_hist            = hist.new.Reg(25, 0.00, 7.0, underflow=True,overflow=True).Weight()  
            mc_hist              = hist.new.Reg(25, 0.00, 7.0, underflow=True,overflow=True).Weight()  
            mc_corr_hist         = hist.new.Reg(25, 0.00, 7.0, underflow=True,overflow=True).Weight()  
        else:
            data_hist            = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight() 
            mc_hist              = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight() 
            mc_corr_hist         = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight() 
                
        weights_forid = weights_forid/torch.sum(weights_forid)

        data_hist.fill(  np.array(data_forid[:,i].cpu() )  )
        mc_hist.fill(    np.array(mc_forid[:,i].cpu() )        , weight = len(data_forid)*weights_forid.cpu() )
        mc_corr_hist.fill( np.array(samples_forid[:,i].cpu())  , weight = len(data_forid)*weights_forid.cpu() )

        try:
            os.makedirs( './general_plots/', exist_ok=True )
        except:
            print( '\nIt was not possible to create the evaluation folder or the folder already exists. - exiting' )

        plott( data_hist , mc_hist, mc_corr_hist , './general_plots/' + str(inputs[i]) +'.pdf', xlabel = str(latex_inputs[i])  )
        print('eh pois e!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Apply a trained NSF flows to HiggsDNA output")
    parser.add_argument('-path'  , '--path'  , type=str, help= "Path to the yaml file that contaings the yaml with the configurations")
    args = parser.parse_args()

    main()
