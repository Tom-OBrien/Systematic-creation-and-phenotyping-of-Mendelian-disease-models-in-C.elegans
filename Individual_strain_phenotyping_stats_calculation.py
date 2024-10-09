#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:34:15 2021
@author: tobrien

This script makes the full phenotyping figures for all strains. It calculates
pairwise stats for all features (using permuation t-tests) and makes blue
light imaging/timeseries plots. 

To make feature plots: stats must either be calculated again or their location
set as the save location for figures. This save location can then be used
to make the number of significant feature plots shown in Figure 2.

**NOTE: This uses random label shuffling to calculate p-values for each feature
re-running the script will therefore result in slight variations of stats from
those saved in the strain stats files folder within the data repository.

"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import chain
from tierpsytools.analysis.significant_features import k_significant_feat
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.drug_screenings.filter_compounds import (
    compounds_with_low_effect_univariate)
from tierpsytools.analysis.statistical_tests import (univariate_tests,
                                                     _multitest_correct)

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
                    make_clustermaps,
                    clustered_barcodes,
                    feature_box_plots,
                    window_errorbar_plots,
                    CUSTOM_STYLE,
                    plot_frac_by_mode, 
                    MODECOLNAMES)
from strain_cmap import full_strain_cmap as STRAIN_cmap

#%% Choose what type of analysis to do
# N2 analysis looks at N2 only data across all the screening days
N2_analysis=True
# All stim calcualtes stats and makes boxplots
# Blue light makes windowed feature plots
# Timeseries plots fraction of worms moving from raw video data
ANALYSIS_TYPE = ['all_stim', 'bluelight','timeseries'] #options:['all_stim','timeseries','bluelight']
# Choose if to plot lineplots of motion modes
motion_modes=True
# Choose if to recalculate stats (several methods are avaliable using the in
# built Tierpsy tools modele, for paper we use permutation t-tests)
do_stats=False

#%% The data was collected in two sets, therefore we load them both
#  Data 1
Data1_FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_1/Data1_FeatureMatrix.csv') 
Data1_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_1/Data1_metadata.csv')
# Data 2
Data2_FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_2/Data2_FeatureMatrix.csv')
Data2_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_2/Data2_metadata.csv')
# Path to timeseries files extracted by Tierpsy
STRAIN_TIME_SERIES = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/Timeseries_of_all_disease_model_strains')
# Set the control strain
CONTROL_STRAIN = 'N2'  
# Put strains already analysed into the list, these will be filtered out
# from reanalysis
strains_done = [
                ]
#%%Setting plotting styles, filtering data & renaming strains
if __name__ == '__main__':
    # Set stats test to be performed and set save directory for output
    which_stat_test = 'permutation_ttest'  # permutation or LMM
    if which_stat_test == 'permutation_ttest':
        # Set the save path for the data here
        figures_dir = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/test/All_individual_strain_statistics')

    # Custom mplt style card to set figure parameters
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    # Read in and filter first dataset
    Data1_featMat = pd.read_csv(Data1_FEAT_FILE, index_col=False)
    Data1_metadata = pd.read_csv(Data1_METADATA_FILE, index_col=False)
    Data1_featMat = filter_nan_inf(Data1_featMat, 0.5, axis=1, verbose=True)
    Data1_metadata = Data1_metadata.loc[Data1_featMat.index]
    Data1_featMat = filter_nan_inf(Data1_featMat, 0.05, axis=0, verbose=True)
    Data1_featMat = Data1_featMat.fillna(Data1_featMat.mean())
    Data1_metadata = Data1_metadata.loc[Data1_featMat.index]
    # filter features
    Data1_feat_df, Data1_meta_df, Data1_featsets = filter_features(Data1_featMat,
                                                                   Data1_metadata)
    
    # %% Do the same for the second set of data
    Data2_featMat = pd.read_csv(Data2_FEAT_FILE, index_col=False)
    Data2_metadata = pd.read_csv(Data2_METADATA_FILE, index_col=False)
    Data2_featMat = filter_nan_inf(Data2_featMat, 0.5, axis=1, verbose=True)
    Data2_metadata = Data2_metadata.loc[Data2_featMat.index]
    Data2_featMat = filter_nan_inf(Data2_featMat, 0.05, axis=0, verbose=True)
    Data2_featMat = Data2_featMat.fillna(Data2_featMat.mean())
    Data2_metadata = Data2_metadata.loc[Data2_featMat.index]
    # filter features
    Data2_feat_df, Data2_meta_df, Data2_featsets = filter_features(
                                                Data2_featMat, Data2_metadata)
    
    #%% Concatenate the two datasets together
    append_feat_df = [Data1_feat_df, Data2_feat_df]
    append_meta_df = [Data1_meta_df, Data2_meta_df]
    
    feat = pd.concat(append_feat_df,
                     axis=0,
                     ignore_index=True)

    meta = pd.concat(append_meta_df,
                     axis=0,
                     ignore_index=True)
    
    feat = pd.DataFrame(feat)
    meta = pd.DataFrame(meta)
    # Saving a copy of the dataframes to work from
    feat_df = feat 
    meta_df = meta 
    featsets = Data2_featsets
    # Set date in a nicer format for plotting
    meta['date_yyyymmdd'] = pd.to_datetime(
        meta['date_yyyymmdd'], format='%Y%m%d').dt.date

    #%% Remove wells annotated as bad
    n_samples = meta.shape[0]
    bad_well_cols = [col for col in meta.columns if 'is_bad' in col]
    bad = meta[bad_well_cols].any(axis=1)
    meta = meta.loc[~bad,:]
    
    #%% Find all the unique genes within the metadata
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    # Remove already analysed genes from the list
    genes = list(set(genes) - set(strains_done))
    genes.sort()
    strain_numbers = []
    # Duplcate date and imaging column to use with helper functions
    imaging_date_yyyymmdd = meta['date_yyyymmdd']
    imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
    meta['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd
    
#%% N2 analysis only- makes cluster maps of N2 features
    # Function to select N2 only from the dataset
    if N2_analysis:
        feat_df, meta_df, idx, gene_list = select_strains(['N2'],
                                                          CONTROL_STRAIN,
                                                          feat_df=feat,
                                                          meta_df=meta)

        feat_df.drop_duplicates(inplace=True)
        meta_df.drop_duplicates(inplace=True)

        # Removes nan's, bad wells, bad days and selected tierpsy features
        feat_df, meta_df, featsets = filter_features(feat_df,
                                                     meta_df)
        
        # Make a stimuli colour map/ look up table with sns
        stim_cmap = sns.color_palette('Pastel1',3)
        stim_lut = dict(zip(STIMULI_ORDER.keys(), stim_cmap))
        feat_lut = {f:v for f in featsets['all'] for k,v in stim_lut.items() if k in f}

        # Impute nans from feature dataframe
        feat_nonan = impute_nan_inf(feat_df)
        # Calculate Z score of features
        featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], 
                                               axis=0),
                             columns=featsets['all'],
                             index=feat_nonan.index)
        # Assert no nans within the data
        assert featZ.isna().sum().sum() == 0
        # Make a clustermap of the N2 only data
        N2clustered_features = make_clustermaps(featZ,
                                                meta_df,
                                                featsets,
                                                strain_lut={'N2': 
                                                            (0.6, 0.6, 0.6)},
                                                feat_lut=feat_lut,
                                                saveto=figures_dir)
        # Write order of clustered features into .txt file - this is used to
        # order future clustermaps within the paper
        for k, v in N2clustered_features.items():
            with open(figures_dir / 'N2_clustered_features_{}.txt'.format(k), 'w+') as fid:
                for line in v:
                    fid.write(line + '\n')
        # If not plotting heatmaps, read cluster features file and make dict
        # for plotting strain heatmaps etc later on in script
    else:
        N2clustered_features = {}
        for fset in STIMULI_ORDER.keys():
            N2clustered_features[fset] = []
            with open(figures_dir / 
                     'N2_clustered_features_{}.txt'.format(fset), 'r') as fid:
                N2clustered_features[fset] = [l.rstrip() 
                                              for l in fid.readlines()]
        with open(figures_dir / 'N2_clustered_features_{}.txt'.format('all'), 'r') as fid:
            N2clustered_features['all'] = [l.rstrip() for l in fid.readlines()]

#%% Counting timer of individual gene selected for analysis
    for count, g in enumerate(genes):
        print('Analysing {} {}/{}'.format(g, count+1, len(genes)))
        candidate_gene = g
        
        # Set save path for figres
        saveto = figures_dir / candidate_gene
        saveto.mkdir(exist_ok=True)
        
        # Make a colour map for control and target strain- Here I use a
        # hardcoded strain cmap to keep all figures consistent for paper
        strain_lut = {}
        candidate_gene_colour = STRAIN_cmap[candidate_gene]

        if 'all_stim' in ANALYSIS_TYPE:
            print ('all stim plots for {}'.format(candidate_gene))

       #Uses Ida's helper to again select individual strain for analysis
            feat_df, meta_df, idx, gene_list = select_strains([candidate_gene],
                                                              CONTROL_STRAIN,
                                                              feat_df=feat,
                                                              meta_df=meta)
       # Filter out unwanted features i.e dorsal feats
            feat_df, meta_df, featsets = filter_features(feat_df,
                                                         meta_df)

            strain_lut, stim_lut, feat_lut = make_colormaps(gene_list,
                                                            featlist=featsets['all'],
                                                            idx=idx,
                                                            candidate_gene=[candidate_gene],
                                                            )
        # Save colour maps as legends/figure keys for use in paper
            plot_colormap(strain_lut)
            plt.savefig(saveto / 'strain_cmap.png')
            plot_cmap_text(strain_lut)
            plt.savefig(saveto / 'strain_cmap_text.png')

            plot_colormap(stim_lut, orientation='horizontal')
            plt.savefig(saveto / 'stim_cmap.png')
            plot_cmap_text(stim_lut)
            plt.savefig(saveto / 'stim_cmap_text.png')

            plt.close('all')

            #%% Impute nan's and calculate Z scores of features for strains
            feat_nonan = impute_nan_inf(feat_df)

            featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], 
                                                   axis=0),
                                 columns=featsets['all'],
                                 index=feat_nonan.index)

            assert featZ.isna().sum().sum() == 0
            #%% Make a nice clustermap of features for strain & N2
            # Plotting helper saves separate cluster maps for: prestim, postim, bluelight and all conditions
            (saveto / 'clustermaps').mkdir(exist_ok=True)

            clustered_features = make_clustermaps(featZ=featZ,
                                                  meta=meta_df,
                                                  featsets=featsets,
                                                  strain_lut=strain_lut,
                                                  feat_lut=feat_lut,
                                                  saveto=saveto / 'clustermaps')
            plt.close('all')
            
            # Make a copy of the cluster map for plotting pVals and selected
            # features later on in this script without overwriting plot
            N2clustered_features_copy = N2clustered_features.copy()

            (saveto / 'heatmaps').mkdir(exist_ok=True)
            (saveto / 'heatmaps_N2ordered').mkdir(exist_ok=True)
            (saveto / 'boxplots').mkdir(exist_ok=True)
            
            # Calculate stats using permutation t-tests or LMM with Tierpsy
            # univariate stats modules
            if do_stats:
                    if which_stat_test == 'permutation_ttest':
                        _, unc_pvals, unc_reject = univariate_tests(
                            feat_nonan, y=meta_df['worm_gene'],
                            control='N2',
                            test='t-test',
                            comparison_type='binary_each_group',
                            multitest_correction=None,
                            n_permutation_test=10000,
                            perm_blocks=meta_df['imaging_date_yyyymmdd'],
                            )
                        reject, pvals = _multitest_correct(
                            unc_pvals, 'fdr_by', 0.05)
                        unc_pvals = unc_pvals.T
                        pvals = pvals.T
                        reject = reject.T
                    
                    # A linear mixmodel can also be used to analyse the data
                    elif which_stat_test == 'LMM':
                        _, _, _, reject, pvals = compounds_with_low_effect_univariate(
                            feat_df, meta_df['worm_gene'],
                            drug_dose=None,
                            random_effect=meta_df['imaging_date_yyyymmdd'],
                            control='N2',
                            test='LMM',
                            comparison_type='binary_each_dose',
                            multitest_method='fdr_by',
                            fdr=0.05,
                            n_jobs=-1
                            )
                    else:
                        raise ValueError((
                            f'Invalid value "{which_stat_test}"'
                            ' for which_stat_test'))
                    # massaging data to be in keeping with downstream analysis
                    assert pvals.shape[0] == 1, 'the output is supposed to be one line only I thought'
                    assert all(reject.columns == pvals.columns)
                    assert reject.shape == pvals.shape
                    # set the pvals over threshold to NaN - These are set to nan for convinence later on
                    bhP_values = pvals.copy(deep=True)
                    bhP_values.loc[:, ~reject.iloc[0, :]] = np.nan
                    bhP_values['worm_gene'] = candidate_gene
                    bhP_values.index = ['p<0.05']

                    # check the right amount of features was set to nan
                    assert reject.sum().sum() == bhP_values.notna().sum().sum()-1
                    
                    # also save the corrected and uncorrected pvalues, without
                    # setting the rejected ones to nan, just keeping the same
                    # dataframe format as bhP_values
                    for p_df in [unc_pvals, pvals]:
                        p_df['worm_gene'] = candidate_gene
                        p_df.index = ['p value']
                    unc_pvals.to_csv(
                        saveto/f'{candidate_gene}_uncorrected_pvals.csv',
                        index=False)
                    pvals.to_csv(
                        saveto/f'{candidate_gene}_fdrby_pvals.csv',
                        index=False)
                    # Save total number of significant feats as .txt file
                    with open(saveto / 'sig_feats.txt', 'w+') as fid:
                        fid.write(str(bhP_values.notna().sum().sum()-1) + ' significant features out of \n')
                        fid.write(str(bhP_values.shape[1]-1))

                    bhP_values.to_csv(saveto / '{}_stats.csv'.format(candidate_gene),
                                      index=False)
                    
                # If not calculating stats, read the .csv file for plotting
            else:
                    bhP_values = pd.read_csv(saveto / '{}_stats.csv'.format(candidate_gene),
                                             index_col=False)
                    bhP_values.rename(mapper={0:'p<0.05'},
                                      inplace=True)
                    
        #%%#I mport features to be plotted from a .txt file and make boxplots
            # Find .txt file (within save directory) and generate list of all feats to plot
            feat_to_plot_fname = list(saveto.rglob('feats_to_plot.txt'))[0]
            selected_feats = []
            with open(feat_to_plot_fname, 'r') as fid:
                    for l in fid.readlines():
                        selected_feats.append(l.rstrip().strip(','))

            all_stim_selected_feats=[]
            for s in selected_feats:
                    all_stim_selected_feats.extend([f for f in featsets['all'] if '_'.join(s.split('_')[:-1])=='_'.join(f.split('_')[:-1])])

                # Make a cluster map of strain vs N2
            clustered_barcodes(clustered_features, selected_feats,
                                    featZ,
                                    meta_df,
                                    bhP_values,
                                    saveto / 'heatmaps')

                # Use the copy of the N2 cluster map (made earlier) and plot
                # cluster map with pVals of all features alongside an asterix
                # denoting the selected features used to make boxplots
            clustered_barcodes(N2clustered_features_copy, selected_feats,
                                    featZ,
                                    meta_df,
                                    bhP_values,
                                    saveto / 'heatmaps_N2ordered')

                # Generate boxplots of selected features containing correct
                # pValues and formatted nicely
            for f in  all_stim_selected_feats:
                    feature_box_plots(f,
                                      feat_df,
                                      meta_df,
                                      strain_lut,
                                      show_raw_data='date',
                                      bhP_values_df=bhP_values
                                      )
                    plt.legend('',frameon=False)
                    plt.tight_layout()
                    plt.savefig(saveto / 'boxplots' / '{}_boxplot.png'.format(f),
                                bbox_inches='tight',
                                dpi=200)
                    plt.close('all')
                
        #%% Using window feature summaries to look at bluelight conditions
        if 'bluelight' in ANALYSIS_TYPE:
            # Set the path for the worm genes collected in the two different
            # datasets based upon what is in the list below
            data1_genes = ['cat-4', 'dys-1', 'pink-1', 'gpb-2', 'kcc-2',
                           'snf-11', 'snn-1', 'unc-25', 'unc-43', 'unc-49',
                           'figo-1']
            
            if candidate_gene in data1_genes:
                WINDOWS_FILES = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_1/window_summaries')
                WINDOWS_METADATA = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_1/window_summaries/window_summaries_metadata.csv')
            
            else:
                WINDOWS_FILES = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_2/window_summaries')
                WINDOWS_METADATA = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_2/window_summaries/Windows_metadata.csv')

            window_files = list(WINDOWS_FILES.rglob('*_window_*'))
            window_feat_files = [f for f in window_files if 'features' in str(f)]
            window_feat_files.sort(key=find_window)
            window_fname_files = [f for f in window_files if 'filenames' in str(f)]
            window_fname_files.sort(key=find_window)
        
            assert (find_window(f[0]) == find_window(f[1]) for f in list(zip(
                window_feat_files, window_fname_files)))
        
        # Use Ida's helper function to read in window files and concat into DF
            feat_windows = []
            meta_windows = []
            for c,f in enumerate(list(zip(window_feat_files, window_fname_files))):
                _feat, _meta = read_disease_data(f[0],
                                                 f[1],
                                                 WINDOWS_METADATA,
                                                 drop_nans=True)
                _meta['window'] = find_window(f[0])
                
                meta_windows.append(_meta)
                feat_windows.append(_feat)
    
            meta_windows = pd.concat(meta_windows)
            meta_windows.reset_index(drop=True,
                                     inplace=True)
            
            feat_windows = pd.concat(feat_windows)
            feat_windows.reset_index(drop=True,
                                 inplace=True)

            
            print ('all window_plots for {}'.format(candidate_gene))
            meta_windows.replace({'C43B7.2':'figo-1'},inplace=True)
            
            # Call dataframes window specific dataframes (made earlier)
            feat_windows_df, meta_windows_df, idx, gene_list = select_strains(
                                                          [candidate_gene],
                                                          CONTROL_STRAIN,
                                                          meta_windows,
                                                          feat_windows)

            # Filter out only the bluelight features
            bluelight_feats = [f for f in feat_windows_df.columns if 'bluelight' in f]
            feat_windows_df = feat_windows_df.loc[:,bluelight_feats]

            feat_windows_df, meta_windows_df, featsets = filter_features(feat_windows_df,
                                                                   meta_windows_df)
            
            bluelight_feats = list(feat_windows_df.columns)
            
            strain_lut_bluelight, stim_lut, feat_lut = make_colormaps(gene_list,
                                                            featlist=bluelight_feats,
                                                            idx=idx,
                                                            candidate_gene=[candidate_gene],
                                                            )

            #%% Fill nans and calculate Zscores of window feats
            feat_nonan = impute_nan_inf(feat_windows_df)

            featZ = pd.DataFrame(data=stats.zscore(feat_nonan[bluelight_feats], axis=0),
                                 columns=bluelight_feats,
                                 index=feat_nonan.index)

            assert featZ.isna().sum().sum() == 0

            #%% Find top significant feats that differentiate between prestim and bluelight
            #make save directory and set layout for plots using dictionary
            (saveto / 'windows_features').mkdir(exist_ok=True)
            meta_windows_df['light'] = [x[1] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
            meta_windows_df['window_sec'] = [x[0] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
            meta_windows_df['stim_number'] = [x[2] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]

            y_classes = ['{}, {}'.format(r.worm_gene, r.light) for i,r in meta_windows_df.iterrows()]

            # Using tierpsytools to find top 100 signifcant feats
            kfeats, scores, support = k_significant_feat(
                    feat_nonan,
                    y_classes,
                    k=100,
                    plot=False,
                    score_func='f_classif')
            
            # Grouping by stimulation number and line making plots for entire
            # experiment and each individual burst window
            stim_groups = meta_windows_df.groupby('stim_number').groups
            for f in kfeats[:50]:
                (saveto / 'windows_features' / f).mkdir(exist_ok=True)
                window_errorbar_plots(feature=f,
                                      feat=feat_windows_df,
                                      meta=meta_windows_df,
                                      cmap_lut=STRAIN_cmap)
                plt.savefig(saveto / 'windows_features' / f / 'allwindows_{}'.format(f), dpi=200)
                plt.close('all')

                for stim,locs in stim_groups.items():
                    window_errorbar_plots(feature=f,
                                          feat=feat_windows_df.loc[locs],
                                          meta=meta_windows_df.loc[locs],
                                          cmap_lut=STRAIN_cmap)
                    plt.savefig(saveto / 'windows_features' / f / 'window{}_{}'.format(stim,f),
                                dpi=200)
                    plt.close('all')

            #%% Calculating motion modes from bluelight features and making 
            # plots of these- saved in a sub-folder within bluelight analysis
            # if motion_modes:
            mm_feats = [f for f in bluelight_feats if 'motion_mode' in f]
            (saveto / 'windows_features' / 'motion_modes').mkdir(exist_ok=True)
            sns.set_style('ticks')
            for f in mm_feats:
                    window_errorbar_plots(feature=f,
                                          feat=feat_windows_df,
                                          meta=meta_windows_df,
                                          cmap_lut=strain_lut)
                    plt.savefig(saveto / 'windows_features' / 'motion_modes' / '{}'.format(f),
                                dpi=200)
                    plt.close('all')
                    for stim,locs in stim_groups.items():
                        window_errorbar_plots(feature=f,
                                              feat=feat_windows_df.loc[locs],
                                              meta=meta_windows_df.loc[locs],
                                              cmap_lut=strain_lut)
                        plt.savefig(saveto / 'windows_features' / 'motion_modes' / 'window{}_{}'.format(stim,f),
                                    dpi=200)
                        plt.close('all')

        #%% Make timerseries plots
        if 'timeseries' in ANALYSIS_TYPE:
             save_ts = saveto / 'timeseries'
             save_ts.mkdir(exist_ok=True)
             print ('timeseries plots for {}'.format(candidate_gene))  
             meta_ts = meta
             keep = [candidate_gene]
             keep.append('N2')
             mask = meta_ts['worm_gene'].isin(keep)
             meta_ts=meta_ts[mask]

             TS_STRAINS = {'plot':  [candidate_gene]}  
            # Make a list of strains with chain function (returns one iterable
            # from a list of several (not strictly necessary for this set of figs)
             ts_strain_list = list(chain(*TS_STRAINS.values()))
            
             # Find .hdf5 files of selected strains from root directory and read in
             # confidence intervals have already been calculated prior this results
             # in a list of 2 dataframes
             timeseries_df = []
             for g in ts_strain_list:
                _timeseries_fname = STRAIN_TIME_SERIES / '{}_timeseries.hdf5'.format(g)
                timeseries_df.append(pd.read_hdf(_timeseries_fname,
                                                  'frac_motion_mode_with_ci'))
                
             strain_lut = {candidate_gene:STRAIN_cmap[candidate_gene],
                          'N2':(0.6,0.6,0.6)}
          
             # Convert the list into one big dataframe and reset index
             timeseries_df = pd.concat(timeseries_df)
             timeseries_df.reset_index(drop=True, inplace=True)
            
            # Select all calculated faction modes for strains of interest and control
             frac_motion_modes = [timeseries_df.query('@ts_strain_list in worm_gene')]
             frac_motion_modes.append(timeseries_df.query('@CONTROL_STRAIN in worm_gene').groupby('timestamp').agg(np.mean))
             frac_motion_modes[1]['worm_gene'] = CONTROL_STRAIN
             frac_motion_modes = pd.concat(frac_motion_modes)
             frac_motion_modes.reset_index(drop=True,inplace=True)
            
            # Plot each of the fraction motion modes as separate plots
            # Modecolnames is just hardcoded list of 'fwd, bckwd and stationary' 
             for m in MODECOLNAMES:
                sns.set_style('ticks')
                plot_frac_by_mode(frac_motion_modes, strain_lut, modecolname=m)
                if m != 'frac_worms_st':
                    plt.ylim([0, 0.5])
                plt.savefig(save_ts / '{}_ts.png'.format(m), dpi=200)
