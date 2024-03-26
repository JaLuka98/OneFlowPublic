import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib.pyplot as pyplot
import torch

import mplhep, hist
plt.style.use([mplhep.style.CMS])
plt.rcParams['text.usetex'] = True
from matplotlib import colors,cm

from sklearn.metrics import roc_curve, auc
import seaborn as sns
import os

import zuko

from matplotlib.legend_handler import HandlerBase

# Custom legend handler to make legend entry invisible
class InvisibleHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        return []


def weighted_quantile(values, quantiles, sample_weight=None):
    """Calculate the weighted quantile of a 1D numpy array.

    Parameters:
    values (numpy.array): data array
    quantiles (float or array-like): quantiles to compute, range from 0.0 to 1.0
    sample_weight (numpy.array, optional): the weights for each value in values

    Returns:
    numpy.array: the computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = sample_weight[sorter]
    weighted_quantiles = np.cumsum(weights_sorted) - 0.5 * weights_sorted
    weighted_quantiles /= np.sum(weights_sorted)
    return np.interp(quantiles, weighted_quantiles, values_sorted)

def find_quantile_for_x(values, cdf, x):
    """Find the quantile of a specific point x in the weighted CDF.
    
    Parameters:
    values (np.array): Sorted data points of the distribution.
    cdf (np.array): Weighted CDF of the distribution.
    x (float): The point to find the quantile for.
    
    Returns:
    float: The quantile of x in the distribution.
    """
    return np.interp(x, values, cdf)

#script that maintaing the plot

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

# Here we study the sizes of the corrections performed by the normalizing flow
# We study here (mc - flow)/mc for the shower shape, where we study the relative size of the corrections
def size_corrections( mc, flow, data, weights, var_names, path):
    
    mc = np.array(mc)
    nl = np.array(flow)

    rel_diff = abs( mc - nl ) #/(mc + 1e-10)
    
    #Make a plot for each one of them
    for i in range( np.shape( mc )[1] ):

        if( 'B' in var_names[i] ):
            bins = np.linspace( 0.0, 3.0, 40 )
        else:
            bins = np.linspace( 0.0, 2.0, 40 )
        
        plt.figure(figsize=(10, 6))
        data_with_overflow = np.where(rel_diff[:,i] > np.max(bins), np.max(bins), rel_diff[:,i])
        if('iso' in var_names[i]):
            plt.hist( rel_diff[:,i], weights=weights ,bins = bins, histtype=u'step',linewidth = 4, color = 'orange' , label= var_names[i])
            plt.yscale('log')
        else:
            plt.hist( data_with_overflow , weights=weights ,bins = bins, histtype=u'step',linewidth = 4, color = 'orange' , label= var_names[i])

        plt.xlabel( 'Distance traveled in ' + str(var_names[i]), fontsize = 28 )
        plt.ylabel( "Normalized to unity" , fontsize = 26 )

        plt.margins(x=0)

        mean = np.mean(rel_diff[:,i])
        std = np.std(rel_diff[:,i])
                
        plt.grid(True)
        plt.tight_layout()

        # Mkdir the path to store the plots
        os.makedirs(path + "/rel_diff/", exist_ok=True)
        plt.savefig( path + "/rel_diff/Dist_Var_" +str(i) + ".pdf" )
        plt.savefig( path + "/rel_diff/Dist_Var_" +str(i) + ".png" )


def ROC_trainig_and_validation_curves(X_train,y_train, X_test, y_test, w_train,w_test,booster, X_train_uncorrected,y_train_uncorrected, X_test_uncorrected, y_test_uncorrected, w_train_uncorrected,w_test_uncorrected,booster_uncorrected , path):
    
    # before correction (nominal MC)
    y_score_train = booster_uncorrected.predict(X_train_uncorrected)
    fpr_train, tpr_train, _ = roc_curve(y_train_uncorrected, y_score_train, sample_weight=w_train_uncorrected)
    roc_auc_train = auc(fpr_train,tpr_train)

    y_score_test = booster_uncorrected.predict(X_test_uncorrected)
    fpr_test, tpr_test, _ = roc_curve(y_test_uncorrected, y_score_test, sample_weight=w_test_uncorrected)
    roc_auc_test = auc(fpr_test,tpr_test)

    # After correcting the mc samples
    y_score2_train = booster.predict(X_train)
    fpr2_train, tpr2_train, _ = roc_curve(y_train, y_score2_train, sample_weight=w_train)
    roc_auc2_train = auc(fpr2_train,tpr2_train)

    y_score2_test = booster.predict(X_test)
    fpr2_test, tpr2_test, _ = roc_curve(y_test, y_score2_test, sample_weight=w_test)
    roc_auc2_test = auc(fpr2_test,tpr2_test)

    plt.figure()
    lw = 2

    ploto, = plt.plot( [-1000], label =  'Test set ROC curves', linestyle="none", color='none')
    
    r1, = plt.plot(fpr_test, tpr_test, color='firebrick',
            lw=lw, label=r'Nominal toy sim. vs. toy data ' % roc_auc_test)
    
    r2, = plt.plot(fpr2_test, tpr2_test, color='darkgreen',
            lw=lw, label=r'Corrected toy sim. vs. toy data ' % roc_auc2_test)    

    # Creating a proxy artist for line2 with no line or marker visible
    invisible_line = plt.Line2D([0], [0], linestyle="none", color='none')

    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    #plt.title('Receiver operating characteristic')
    #leg = plt.legend( [ploto,r1,r2],['Test set ROC curves',r'Nominal toy sim. vs. toy data (area = %0.2f)' % roc_auc_test, r'Corrected toy sim. vs. toy data (area = %0.2f)' % roc_auc2_test ],loc="lower right",fontsize = 19)
    plt.savefig( path + 'Schon_ROC_curves.pdf' )
    
# function to calculate the weighted profiles quantiles!
def weighted_quantiles_interpolate(values, weights, quantiles=0.5):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]

# Calculates the weighted median stat errors for the profile plots 
# more details on: https://stats.stackexchange.com/questions/59838/standard-error-of-the-median/61759#61759

def weighted_quantiles_std(values, weights, quantiles=0.5):
    
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    
    n_events = [i[np.searchsorted(c, np.array(quantiles) * c[-1])]]
    events = values[:int(n_events[0])]

    # Ensure weights sum to 1 (normalize if they don't)
    w_normalized = weights[i][:int(n_events[0])] / np.sum(weights[i][:int(n_events[0])])

    # Calculate weighted mean
    weighted_mean = np.sum(w_normalized * events)

    # Calculate weighted variance
    weighted_variance = np.sum(w_normalized * (events - weighted_mean)**2)

    # Calculate weighted standard deviation
    weighted_std = np.sqrt(weighted_variance)

    error = 1.253*weighted_std/np.sqrt(len(events))
    return error

#plot the diference in correlations betwenn mc and data and data and flow
def plot_correlation_matrices(data,mc,mc_corrected, mc_weights, var_names, path):

    #calculating the covariance matrix of the pytorch tensors
    data_corr         = torch.cov( data.T         , )
    mc_corr           = torch.cov( mc.T           , aweights = torch.Tensor( mc_weights  ))
    mc_corrected_corr = torch.cov( mc_corrected.T , aweights = torch.Tensor( mc_weights  ))

    #from covariance to correlation matrices
    data_corr         = torch.inverse( torch.sqrt( torch.diag_embed( torch.diag(data_corr))) ) @ data_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(data_corr))) ) 
    mc_corr           = torch.inverse( torch.sqrt( torch.diag_embed(torch.diag(mc_corr)) )) @ mc_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corr))) ) 
    mc_corrected_corr = torch.inverse( torch.diag_embed(torch.sqrt(torch.diag(mc_corrected_corr)) )) @ mc_corrected_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corrected_corr))) ) 

    # matrices setup ended! Now plotting part!
    fig, ax = plt.subplots(figsize=(41,41))
    cax = ax.matshow( 100*( data_corr - mc_corrected_corr ), cmap = 'bwr', vmin = -35, vmax = 35)
    cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize = 70)
    cbar.set_label(r'Difference in correlation coefficient $[\%]$', rotation=90, loc = 'center', fontsize = 110, labelpad=60)

    # ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate( 100*( data_corr - mc_corrected_corr )):
        mean = mean + abs(z)
        count = count + 1
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 85)    
    
    ax.yaxis.labelpad = 20
    ax.xaxis.labelpad = 20
    mean = mean/count
    #ax.set_xlabel(r'$100 \cdot (Corr^{Data}[X_{i},X_{J}] - Corr^{Simulation^{Corr}}[X_{i},X_{J}]) $ ' , loc = 'center' ,fontsize = 100, labelpad=40)
    plt.title( r'$\rho$(toy data) - $\rho$(corrected toy simulation)', fontweight='bold', fontsize = 130 , pad = 60 )
    
    ax.set_xticks(np.arange(len(var_names)))
    ax.set_yticks(np.arange(len(var_names)))
    
    ax.set_xticklabels(var_names, fontsize = 90 , rotation=90 )
    ax.set_yticklabels(var_names, fontsize = 90 , rotation=0  )

    ax.tick_params(axis='both', which='major', pad=30)
    plt.tight_layout()

    plt.savefig(path + '/correlation_matrix_corrected.pdf')

    ####################################
    # Nominal MC vs Data
    #####################################
    fig, ax = plt.subplots(figsize=(41,41))
    cax = ax.matshow( 100*( data_corr - mc_corr ), cmap = 'bwr', vmin = -35, vmax = 35)
    cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize = 90)
    cbar.set_label(r'Difference in correlation coefficient $[\%]$', rotation=90, loc = 'center', fontsize = 110,labelpad=60)
    
    #ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate( 100*( data_corr - mc_corr )):
        mean = mean + abs(z)
        count = count + 1
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 85)    
    
    mean = mean/count
    #ax.set_xlabel(r'$100 \cdot  (Corr^{Data}[X_{i},X_{J}] - Corr^{Simulation}[X_{i},X_{J}]) $ ' , loc = 'center' ,fontsize = 100, labelpad=40)
    plt.title( r'$\rho$(toy data) - $\rho$(nominal toy simulation)',fontweight='bold', fontsize = 140 , pad = 60 )
    
    ax.set_xticks(np.arange(len(var_names)))
    ax.set_yticks(np.arange(len(var_names)))
    
    ax.set_xticklabels(var_names,fontsize = 90 ,rotation=90)
    ax.set_yticklabels(var_names,fontsize = 90 ,rotation=0)

    ax.tick_params(axis='both', which='major', pad=30)
    plt.tight_layout()

    plt.savefig(path + '/correlation_matrix_nominal.pdf')

def calculate_bins_position(array, num_bins=8):

    array_sorted = np.sort(array)  # Ensure the array is sorted
    n = len(array)
    
    # Calculate the exact number of elements per bin
    elements_per_bin = n // num_bins
    
    # Adjust bin_indices to accommodate for numpy's 0-indexing and avoid out-of-bounds access
    bin_indices = [i*elements_per_bin for i in range(1, num_bins)]
    bin_indices.append(n-1)  # Ensure the last index is included for the last bin
    
    # Find the array values at these adjusted indices
    bin_edges = array_sorted[bin_indices]

    bin_edges = np.insert(bin_edges, 0, 0)
    
    return bin_edges

def pt_profile_again(predictions, test, labels, path_to_plot, mc_weights, var='pt'):
    
    # Calculating the profile of the pt variable
    # Lets bin in bins of 0.5 pt
    var_list = ["pt","eta","noise"]
    for var in var_list:

        data_mean    , flow_mean   , position = [],[],[]
        data_mean_q25, flow_mean_q25   = [], []
        data_mean_q75, flow_mean_q75   = [], []

        flow_error, data_error = [],[]
        flow_error_q25, data_error_q25 = [],[]
        flow_error_q75, data_error_q75 = [],[]

        if( var == 'pt' ):
            pt = test[:,4]

            # Using same number of events bins
            sort_index = np.argsort( test[:,4] )
            pt_          = test[:,4][sort_index]

            # lets try to separate the events 
            pt_bins = calculate_bins_position( pt_ )
            np.append( pt_bins, 100 )

        elif( var == 'eta'):
            pt = test[:,5]

            # Using same number of events bins
            sort_index = np.argsort( test[:,5] )
            pt_          = test[:,5][sort_index]

            # lets try to separate the events 
            pt_bins = calculate_bins_position( pt_ )
            pt_bins = pt_bins[:-1] # = 40
            pt_bins = np.append(pt_bins, 400 )

        else:

            pt = test[:,6]            
            sort_index = np.argsort( test[:,6] )
            pt_          = test[:,6][sort_index]

            # lets try to separate the events 
            pt_bins = calculate_bins_position( pt_ )
        
        xerr_ = []
        for i in range( len(pt_bins) - 1 ):

            mask_pt_ = np.logical_and( pt >  pt_bins[i] , pt < pt_bins[i+1]  )

            predictions[mask_pt_][np.array(labels[mask_pt_], dtype=bool)]
            predictions[mask_pt_][~np.array(labels[mask_pt_], dtype=bool)]

            data_mean.append( weighted_quantiles_interpolate( predictions[mask_pt_][np.array(labels[mask_pt_], dtype=bool)]  , weights = np.ones(  len( predictions[mask_pt_][ np.array(labels[mask_pt_], dtype=bool) ] ) )     , quantiles = 0.5))
            flow_mean.append( weighted_quantiles_interpolate( predictions[mask_pt_][~np.array(labels[mask_pt_], dtype=bool)] , weights = mc_weights[mask_pt_][ ~np.array(labels[mask_pt_], dtype=bool) ]                        , quantiles = 0.5 ))
            
            flow_error.append(  weighted_quantiles_std( predictions[mask_pt_][np.array(labels[mask_pt_], dtype=bool)]  , weights = np.ones(  len( predictions[mask_pt_][ np.array(labels[mask_pt_], dtype=bool) ] ) )     , quantiles = 0.5) )
            data_error.append(  weighted_quantiles_std( predictions[mask_pt_][np.array(labels[mask_pt_], dtype=bool)]  , weights = np.ones(  len( predictions[mask_pt_][ np.array(labels[mask_pt_], dtype=bool) ] ) )     , quantiles = 0.5)  )

            # 75th quantile
            data_mean_q75.append( weighted_quantiles_interpolate( predictions[mask_pt_][np.array(labels[mask_pt_], dtype=bool)]  , weights = np.ones(  len( predictions[mask_pt_][ np.array(labels[mask_pt_], dtype=bool) ] ) )     , quantiles = 0.75))
            flow_mean_q75.append( weighted_quantiles_interpolate( predictions[mask_pt_][~np.array(labels[mask_pt_], dtype=bool)] , weights = mc_weights[mask_pt_][ ~np.array(labels[mask_pt_], dtype=bool) ]                        , quantiles = 0.75 ))
            flow_error_q75.append(  weighted_quantiles_std( predictions[mask_pt_][np.array(labels[mask_pt_], dtype=bool)]  , weights = np.ones(  len( predictions[mask_pt_][ np.array(labels[mask_pt_], dtype=bool) ] ) )     , quantiles = 0.75) )
            data_error_q75.append(  weighted_quantiles_std( predictions[mask_pt_][np.array(labels[mask_pt_], dtype=bool)]  , weights = np.ones(  len( predictions[mask_pt_][ np.array(labels[mask_pt_], dtype=bool) ] ) )     , quantiles = 0.75)  )

            # 25th quantile
            data_mean_q25.append( weighted_quantiles_interpolate( predictions[mask_pt_][np.array(labels[mask_pt_], dtype=bool)]  , weights = np.ones(  len( predictions[mask_pt_][ np.array(labels[mask_pt_], dtype=bool) ] ) )     , quantiles = 0.25))
            flow_mean_q25.append( weighted_quantiles_interpolate( predictions[mask_pt_][~np.array(labels[mask_pt_], dtype=bool)] , weights = mc_weights[mask_pt_][ ~np.array(labels[mask_pt_], dtype=bool) ]                        , quantiles = 0.25))

            flow_error_q25.append(  weighted_quantiles_std( predictions[mask_pt_][np.array(labels[mask_pt_], dtype=bool)]  , weights = np.ones(  len( predictions[mask_pt_][ np.array(labels[mask_pt_], dtype=bool) ] ) )     , quantiles = 0.25))
            data_error_q25.append(  weighted_quantiles_std( predictions[mask_pt_][np.array(labels[mask_pt_], dtype=bool)]  , weights = np.ones(  len( predictions[mask_pt_][ np.array(labels[mask_pt_], dtype=bool) ] ) )     , quantiles = 0.25))

            position.append(pt_bins[i] + (pt_bins[i+1] - pt_bins[i] )/2.0)
            xerr_.append( pt_bins[i+1] - pt_bins[i] )

        #ploting the mean as a function of pt
        plt.figure(figsize=(10, 6))

        # 50th quantile
        plt.plot( position , flow_mean , linewidth  = 2 , color = 'red'  , label = r'Corrected toy simulation (median and $25\%/75\%$ quantiles)'    )
        plt.plot( position , data_mean , linewidth = 2 , color = 'green' , label = r'Toy data (median and $25\%/75\%$ quantiles)'           )
            
        # Calculating the upper and lower bounds of the error
        upper_bound = np.array(flow_mean) + np.array(flow_error)
        lower_bound = np.array(flow_mean) - np.array(flow_error)
        plt.fill_between(position, lower_bound, upper_bound, facecolor="none", edgecolor="tab:red", alpha=1, hatch='XXX')

        upper_bound = np.array(data_mean) + np.array(data_error)
        lower_bound = np.array(data_mean) - np.array(data_error)
        plt.fill_between(position, lower_bound, upper_bound, facecolor="none", edgecolor="tab:green", alpha=1, hatch='XXX',)
            
        # 75th quantile
        plt.plot( position , flow_mean_q25 , linewidth  = 2 , linestyle='dashed' , color = 'red'    )
        plt.plot( position , data_mean_q25 , linewidth = 2  , linestyle='dashed' , color = 'green'  )

        plt.plot( position , flow_mean_q75 , linewidth  = 2 , linestyle='dashed' , color = 'red'    )
        plt.plot( position , data_mean_q75 , linewidth = 2  , linestyle='dashed' , color = 'green'  )

        # Calculating the upper and lower bounds of the error
        upper_bound = np.array(flow_mean_q75) + np.array(flow_error_q75)
        lower_bound = np.array(flow_mean_q75) - np.array(flow_error_q75)
        plt.fill_between(position, lower_bound, upper_bound, facecolor="none", edgecolor="tab:red", alpha=1, hatch='XXX')

        upper_bound = np.array(data_mean_q75) + np.array(data_error_q75)
        lower_bound = np.array(data_mean_q75) - np.array(data_error_q75)
        plt.fill_between(position, lower_bound, upper_bound, facecolor="none", edgecolor="tab:green", alpha=1, hatch='XXX')

        # 25th quantile
        upper_bound = np.array(flow_mean_q25) + np.array(flow_error_q25)
        lower_bound = np.array(flow_mean_q25) - np.array(flow_error_q25)
        plt.fill_between(position, lower_bound, upper_bound, facecolor="none",edgecolor="tab:red", alpha=1, hatch='XXX')

        upper_bound = np.array(data_mean_q25) + np.array(data_error_q25)
        lower_bound = np.array(data_mean_q25) - np.array(data_error_q25)
        plt.fill_between(position, lower_bound, upper_bound, facecolor="none", edgecolor="tab:green", alpha=1, hatch='XXX')

        #################################
        plt.ylabel( 'BDT score' )
        plt.legend(fontsize=18, loc = 'upper right')
        plt.ylim( 0.47 , 0.53 )
        plt.tight_layout()
        
        if( var == 'pt' ):
            plt.xlim( 0.0 , 3.0 )
            plt.xlabel( r'$p_{T}$' )
            plt.savefig( path_to_plot + 'BDT_profile_test_pt.pdf' ) 
        elif( var == 'eta'):
            plt.xlim( 0.0 , 2.0 )
            plt.xlabel( r'$\eta$' )
            plt.savefig( path_to_plot + 'BDT_profile_test_eta.pdf' ) 
        else:
            plt.legend(fontsize=18, loc = 'upper center')
            plt.ylim( 0.47 , 0.53 )
            plt.tight_layout()
            plt.xlim( 0.0 , 3.0 )
            plt.xlabel( r'$N$' )            
            plt.savefig( path_to_plot + 'BDT_profile_test_noise.pdf' ) 

def plot_BDT_output(predictions,  test, path_to_plot, trainset = False):
    
    # plot signal and background separately
    plt.figure()

    signal_mean = np.mean( predictions[test.get_label().astype(bool)] )
    flow_mean   = np.mean( predictions[~(test.get_label().astype(bool))] )

    plt.hist(predictions[test.get_label().astype(bool)]   ,bins=np.linspace(0.44,0.56,51),
                histtype='step',color='midnightblue', linewidth = 3, label='Toy data: ' +  str( round(signal_mean,4) ), density = True)
    plt.hist(predictions[~(test.get_label().astype(bool))],bins=np.linspace(0.44,0.56,51),
                histtype='step',color='firebrick',linewidth = 3,label='Corrected toy simulation: ' + str( round(flow_mean,4) ), density = True)

    # make the plot readable
    plt.xlabel('BDT score',fontsize=28)
    plt.ylabel('Fraction of events/Bin width',fontsize=25)
    plt.legend(frameon=False)   
    plt.tight_layout()

    plt.margins(y=0.2)
    plt.margins(x=0.0)

    if( trainset  ):
        plt.savefig( path_to_plot + 'BDT_outputs_trainset.pdf' ) 
    else:
        plt.savefig( path_to_plot + 'BDT_outputs.pdf' ) 

def plot_model_loss(evals_result, plot_path):
    
    train_losses = evals_result['train']['error']
    val_losses = evals_result['validation']['error']

    fig, ax = pyplot.subplots()
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Test')
    ax.legend()
    plt.savefig( plot_path + 'loss_curve.png')


