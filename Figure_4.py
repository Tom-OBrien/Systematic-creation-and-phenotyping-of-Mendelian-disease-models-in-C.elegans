#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 5 Aug 14:32:18 2021

This script makes the NACLN mutant figures shown in Fig.4 of the paper

@author: tobrien

"""
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import chain
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf

sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/Code/Helper_Functions_and_Scripts')
from helper import (read_disease_data,
                    select_strains,
                    filter_features,
                    make_colormaps,
                    find_window,
                    BLUELIGHT_WINDOW_DICT,
                    STIMULI_ORDER, 
                    plot_colormap,
                    plot_cmap_text,
                    feature_box_plots,
                    window_errorbar_plots,
                    CUSTOM_STYLE,
                    make_heatmap_df, 
                    make_barcode, 
                    MODECOLNAMES,
                    short_plot_frac_by_mode)
#%% Update the global paths below using the files within the Zenodo folder
# All these strains were colelcted in the second dataset (Data 2)
ROOT_DIR = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_2')
FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_2/Data2_FeatureMatrix.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_2/Data2_metadata.csv')
WINDOW_FILES = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_2/window_summaries')
WINDOWS_METADATA = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_2/window_summaries/Windows_metadata.csv')
RAW_DATA_DIR = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/Timeseries_of_all_disease_model_strains')

# Make a save directory
saveto = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/test')
saveto.mkdir(exist_ok=True)
# Select the features we want to plot. There are the ones used in the paper
# However, any feature extracted by Tierpsy could be used
FAINTER_FEATURES = [
                    'length_50th_prestim',
                    'speed_neck_90th_bluelight',
                    'motion_mode_paused_fraction_bluelight',
                    'curvature_mean_head_norm_abs_50th_bluelight',
                    ]

# Here I'm chosing some motion modes to look at
FAINTER_BLUELIGHT = [
                     'motion_mode_paused_duration_50th_bluelight',
                     'motion_mode_backward_fraction_bluelight',
                     'motion_mode_backward_duration_50th_bluelight',
                     'motion_mode_forward_fraction_bluelight',
                     ]

# Choose which strains to analyse, and how to group for plotting
STRAINS = ['nca-2',
           'unc-77',
           'unc-80']
# Set control strain
CONTROL_STRAIN = 'N2'

#%% Import data and filter accordingly:
if __name__ == '__main__':

    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
 
    feat = pd.read_csv(FEAT_FILE, index_col=False)
    meta = pd.read_csv(METADATA_FILE, index_col=False)

    
    # Select just date (remove time) for the purposes of nice plotting
    meta['date_yyyymmdd'] = pd.to_datetime(
    meta['date_yyyymmdd'], format='%Y%m%d').dt.date
    meta['imaging_date_yyyymmdd'] = pd.to_datetime(
    meta['imaging_date_yyyymmdd'], format='%Y%m%d').dt.date

    window_files = list(WINDOW_FILES.rglob('*_window_*'))
    window_feat_files = [f for f in window_files if 'features' in str(f)]
    window_feat_files.sort(key=find_window)
    window_fname_files = [f for f in window_files if 'filenames' in str(f)]
    window_fname_files.sort(key=find_window)

    assert (find_window(f[0]) == find_window(f[1]) for f in list(zip(window_feat_files, window_fname_files)))
    
    #%%
    # Make a list of unique genes from combined metadata df that are different
    # to the control and re-name them accordingly
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]

    # Uses Ida's helper function to select candidate genes and select only
    # the data associated with these and the control for analysis
    feat_df, meta_df, idx, gene_list = select_strains(candidate_gene=STRAINS,
                                                    control_strain=CONTROL_STRAIN,
                                                    feat_df=feat,
                                                    meta_df=meta)
    
    # Wrapper function that performs the following:
        # remove wells annotated as bad
        # remove features and wells with too many nans and std=0
        # removes path curvature features
        # makes a feature list of 'all' feature set, i.e. pre/post-stim & bluelight
    feat_df, meta_df, featsets = filter_features(feat_df,
                                                 meta_df)
    
    # Ida's helper function that makes a strain look up table of the selected
    # strains and also makes a unique colour for these (orded by complete 
    # strain list), also makes colour map for the stimuli
    # This function relies on 'idx' calculated by the select strains function
    strain_lut, stim_lut, feat_lut = make_colormaps(gene_list,
                                                    featlist=featsets['all'],
                                                    idx=idx,
                                                    candidate_gene=STRAINS
                                                    )
    
    
    # Save these colour maps in two different formats
    plot_colormap(strain_lut)
    plt.savefig(saveto / 'strain_cmap.png')
    plot_cmap_text(strain_lut)
    plt.savefig(saveto / 'strain_cmap_text.png')

    plot_colormap(stim_lut, orientation='horizontal')
    plt.savefig(saveto / 'stim_cmap.png')
    plot_cmap_text(stim_lut)
    plt.savefig(saveto / 'stim_cmap_text.png')

    plt.close('all')
    
    strain_lut = {'N2': ('lightgrey'),
                  'nca-2':('palegreen'),
                  'unc-77':('lightskyblue'),
                  'unc-80':('coral')}

    #%% 
    # Make nice box plots of the selected features
    for f in  FAINTER_FEATURES:
        feature_box_plots(f,
                          feat_df,
                          meta_df,
                          strain_lut,
                          show_raw_data='date',
                          add_stats=False)
        (saveto / 'boxplots').mkdir(exist_ok=True)
        plt.savefig(saveto / 'boxplots' /'{}_boxplot.pdf'.format(f),
                    bbox_inches='tight',
                    dpi=200)
    plt.close('all')
    
    #%% plot a heatmap/barcode
    
    # Impute nans with tierpsy tools function
    feat_nonan = impute_nan_inf(feat_df)
    
    # Calculate Z-scores of features and save as data frame
    # featsets is an output of the filter_features function
    featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], axis=0),
                         columns=featsets['all'],
                         index=feat_nonan.index)
    
    # Double check no nan's in the Z-score feature matrix
    assert featZ.isna().sum().sum() == 0    
    
    # N2_clustered_features.txt is a file I have already made during the overall
    # analysis of data. Contains how features are ordered according to z-score
    # this keeps the order of these features consistent across plots
    
    # Find N2clustered feats file 
    N2clustered_features = {}
    for fset in STIMULI_ORDER.keys():
        # N2clustered_features[fset] = []
        with open(ROOT_DIR /  'N2_clustered_features_{}.txt'.format(fset), 'r') as fid:
            N2clustered_features[fset] = [l.rstrip() for l in fid.readlines()]
            # NB: rstrip just removes trailing characters from lines
            
    with open(ROOT_DIR / 'N2_clustered_features_{}.txt'.format('all'), 'r') as fid:
        N2clustered_features['all'] = [l.rstrip() for l in fid.readlines()]

    N2clustered_features_copy = N2clustered_features.copy()
    (saveto / 'heatmaps').mkdir(exist_ok=True)
        
    for stim,fset in featsets.items():
        heatmap_df = make_heatmap_df(N2clustered_features_copy[stim],
                                     featZ[fset],
                                     meta_df)
        
        make_barcode(heatmap_df,
                     FAINTER_FEATURES,
                     cm=['inferno']*(heatmap_df.shape[0]-1)+['Pastel1'],
                     vmin_max=[(-1.5,1.5)]*(heatmap_df.shape[0]-1)+[(1,3)])

        plt.savefig(saveto / 'heatmaps' / '{}_heatmap.png'.format(stim))

    
    #%% plot for the windows
    # Make 2 empty lists for data
    feat_windows = []
    meta_windows = []
    
    # For loop that iterates over feat and fname files, obtaining index with
    # enumerate,list these and get the elements of list with zip
    # read_disease data is a wrapper function that:
        # Reads in data with tierpsy tools
        # Aligns by bluelight with tierpsy tools
        # Removes nans with tierpsy tools
    for c,f in enumerate(list(zip(window_feat_files, window_fname_files))):
        _feat, _meta = read_disease_data(f[0],
                                         f[1],
                                         WINDOWS_METADATA,
                                         drop_nans=True)
        _meta['window'] = find_window(f[0])
        
        meta_windows.append(_meta)
        feat_windows.append(_feat)
    
    # Covert list indicies into integers and concat metadata  
    meta_windows = pd.concat(meta_windows)
    meta_windows.reset_index(drop=True,
                             inplace=True)
    meta_windows.worm_gene.replace({'C43B7.2':'figo-1'},
                                   inplace=True)
    
    # Extract date only from metadata
    meta_windows['date_yyyymmdd'] = pd.to_datetime(
    meta_windows['date_yyyymmdd'], format='%Y%m%d').dt.date
    
    feat_windows = pd.concat(feat_windows)
    feat_windows.reset_index(drop=True,
                             inplace=True)
    
     # Again use the select strains helper function to choose the strains of interest   
    feat_windows_df, meta_windows_df, idx, gene_list = select_strains(STRAINS,
                                                  CONTROL_STRAIN,
                                                  meta_windows,
                                                  feat_windows)

    # Search for blue_light features only and make df of these only
    bluelight_feats = [f for f in feat_windows_df.columns if 'bluelight' in f]
    feat_windows_df = feat_windows_df.loc[:,bluelight_feats]
    
    # Same wrapper function as before that performs the following:
        # remove wells annotated as bad
        # remove features and wells with too many nans and std=0
        # removes path curvature features
    feat_windows_df, meta_windows_df, featsets = filter_features(feat_windows_df,
                                                           meta_windows_df)
    
    # Make a list of all the bluelight features 
    bluelight_feats = list(feat_windows_df.columns)

    # Set save to
    (saveto / 'windows_features').mkdir(exist_ok=True)
    # Use the hardcoded bluelight window dictionary to append information about
    # light condition, time and stimuli number to the metadata dataframe
    meta_windows_df['light'] = [x[1] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
    meta_windows_df['window_sec'] = [x[0] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
    meta_windows_df['stim_number'] = [x[2] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]

    # Group by stimul number
    stim_groups = meta_windows_df.groupby('stim_number').groups    
    #%% Plotting bluelight features
    
    window_strain_lut = {'N2': ('grey'),
                  'nca-2':('palegreen'),
                  'unc-77':('lightskyblue'),
                  'unc-80':('coral')}
    
    # Iterate over selected features and make a line plot of thesee
    # First plot all 3 stimuli windows
    for f in FAINTER_BLUELIGHT:
        sns.set_style('ticks')
        (saveto / 'windows_features' / f).mkdir(exist_ok=True)
        window_errorbar_plots(f,
                              feat_windows_df,
                              meta_windows_df,
                              window_strain_lut,
                              plot_legend=True)
        plt.savefig(saveto / 'windows_features' / f / 'allwindows_{}'.format(f), 
                    bbox_inches='tight', dpi=200)
        plt.close('all')

    # Now plot each window separately based upon stimuli/burst number grouping
        for stim,locs in stim_groups.items():
            window_errorbar_plots(f,
                                  feat_windows_df.loc[locs],
                                  meta_windows_df.loc[locs],
                                  window_strain_lut,
                                  plot_legend=True)
            plt.savefig(saveto / 'windows_features' / f / 'window{}_{}.pdf'.format(stim,f),
                        bbox_inches='tight', dpi=200)
            plt.close('all')
    #%% Timeseries plots
    
    # Make a dictionary of strains (the way I have set this up means I can
    # subset groups of strains together to be plotted on separate plots)
    TS_STRAINS = {'fainter':  ['nca-2',
                                'unc-77',
                                'unc-80']}  
    # Make a list of strains with chain function (returns one iterable
    # from a list of several (not strictly necessary for this set of figs)
    ts_strain_list = list(chain(*TS_STRAINS.values()))
    
    # Find .hdf5 files of selected strains from root directory and read in
    # confidence intervals have already been calculated prior this results
    # in a list of 2 dataframes
    timeseries_df = []
    for g in ts_strain_list:
        _timeseries_fname = RAW_DATA_DIR / '{}_timeseries.hdf5'.format(g)
        timeseries_df.append(pd.read_hdf(_timeseries_fname,
                                          'frac_motion_mode_with_ci'))
  
    # Convert the list into one big dataframe and reset index
    timeseries_df = pd.concat(timeseries_df)
    timeseries_df.reset_index(drop=True, inplace=True)
    # Here I only want to plot the first 160 seconds of timeseries data (only 
    # first pulse of blue light) 
    time_drop = timeseries_df['time_s']>160
    timeseries_df = timeseries_df.loc[~time_drop,:]
 
    # Select all calculated faction modes for strains of interest and control
    frac_motion_modes = [timeseries_df.query('@ts_strain_list in worm_gene')]
    frac_motion_modes.append(timeseries_df.query('@CONTROL_STRAIN in worm_gene').groupby('timestamp').agg(np.mean))
    frac_motion_modes[1]['worm_gene'] = CONTROL_STRAIN
    frac_motion_modes = pd.concat(frac_motion_modes)
    frac_motion_modes.reset_index(drop=True,inplace=True)
    
    # Plot each of the fraction motion modes as separate plots
    # Modecolnames is just hardcoded list of 'fwd, bckwd and stationary' 
    (saveto / 'first_stimuli_ts').mkdir(exist_ok=True)
    for m in MODECOLNAMES:
        sns.set_style('ticks')
        short_plot_frac_by_mode(frac_motion_modes, strain_lut, modecolname=m)
        if m != 'frac_worms_st':
            plt.ylim([0, 0.5])
        plt.savefig(saveto / 'first_stimuli_ts' /'{}_ts.png'.format(m), dpi=200)
                        