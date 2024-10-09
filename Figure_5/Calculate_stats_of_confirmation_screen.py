#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:33:31 2023

This script calculates pairwise stats of treated unc-80 vs the untreated N2.
These statstics are used to calculate the number of rescued features vs the
number of side effects shown in Figure 5C and determine which compound has
the best rescue

**NOTE: statstics are calcualted using permutation tests that randomly shuffle
labels. Hence, re-running this code from the same data always results in slight
variations in the total number of significant features, and the p-values
calculated**

@author: tobrien
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.read_data.hydra_metadata import (align_bluelight_conditions,
                                                   read_hydra_metadata)
from tierpsytools.analysis.statistical_tests import (univariate_tests,
                                                     _multitest_correct)
sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/Code/Helper_Functions_and_Scripts')
from helper import (select_strains,
                    filter_features,
                    make_colormaps, 
                    feature_box_plots,
                    CUSTOM_STYLE)
#%% Set paths for data and parameters for running the script
ANALYSIS_TYPE = ['all_stim'] 
do_stats=True
filter_wells=True
# Define file locations, save directory and control strain
FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/ConfirmationScreen/features.csv') 
FNAME_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/ConfirmationScreen/filenames.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/ConfirmationScreen/metadata.csv')
# Any feature can be plotted if passed as .txt file
feats_plot = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/Code/Figure_5/selected_feats')
# Set the control (target) strain to compare each treatment to
CONTROL_STRAIN = 'N2'  
# Strains already analysed (removes them from re-analysis) e.g. unc-80 control
strains_done = ['unc-80']
#%%Setting plotting styles, filtering data & renaming strains
if __name__ == '__main__':
    # Set stats test to be performed and set save directory for output
    which_stat_test = 'permutation_ttest'  # permutation or LMM
    if which_stat_test == 'permutation_ttest':
        figures_dir = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/ConfirmationScreen/Stats_of_each_compound_vs_untreated_N2')

    # CUSTOM_STYLE= mplt style card ensuring figure style remains consistent
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    
    # Read in data and align by bluelight with tierpsy tools functions
    feat, meta = read_hydra_metadata(
        FEAT_FILE,
        FNAME_FILE,
        METADATA_FILE)
    feat, meta = align_bluelight_conditions(feat, meta, how='inner')

    # Converting metadata date into nicer format when plotting
    meta['date_yyyymmdd'] = pd.to_datetime(
        meta['date_yyyymmdd'], format='%Y%m%d').dt.date
    
    # Filter out nan's within specified columns and print .csv of these    
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'date_yyyymmdd']]
    nan_worms.to_csv(
        METADATA_FILE.parent / 'nan_worms.csv', index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)            

    if filter_wells==True:
        mask = meta['well_label'].isin([1.0, 3.0])
        
    meta = meta[mask]    
    feat = feat[mask]

    # Combine information about strain and drug treatment
    meta['analysis'] = meta['worm_gene'] + '+' + meta['drug_type']
    # Rename controls for ease of use
    
    # Drop water only controls- focus on solvent only (DMSO) controls
    mask = meta['analysis'].isin(['unc-80+water', 'N2+water',
                                  'unc-80+no compound', 'N2+no compound'])
    meta = meta[~mask]
    feat = feat[~mask]
    
    # Some of the compound names have their brand name in the metadata, 
    # simply rename these to  help plots fit onto axes
    meta['analysis'].replace({
        'unc-80+Atorvastatin calcium (Lipitor)':'unc-80+Atorvastatin calcium',
        'unc-80+Daunorubicin HCl (Daunomycin HCl)':'unc-80+Daunorubicin HCl',
        'unc-80+Ciprofloxacin (Cipro)':'unc-80+Ciprofloxacin',
        'unc-80+Ivabradine HCl (Procoralan)':'unc-80+Ivabradine HCl',
        'unc-80+Iloperidone (Fanapt)':'unc-80+Iloperidone',
        'unc-80+Clozapine (Clozaril)':'unc-80+Clozapine',
        'unc-80+Mitotane (Lysodren)':'unc-80+Mitotane',
        'unc-80+Abitrexate (Methotrexate)': 'unc-80+Abitrexate',
        'unc-80+Sulindac (Clinoril)':'unc-80+Sulindac',
        'unc-80+Mesalamine (Lialda)':'unc-80+Mesalamine',
        'unc-80+Fenofibrate (Tricor, Trilipix)':'unc-80+Fenofibrate'
                            },
                             inplace=True)       
    
        # To make drug names easier to read, I'm simply stripping the first few characters
    meta['analysis'] = meta['analysis'].map(lambda x: x.lstrip('unc-80+'))
    # The method above renames unc-80 to 'DMSO' so just rename this here
    meta['analysis'].replace({'DMSO':'unc-80',
                              'N2+DMSO':'N2'},
                            inplace=True)
    
    # Update worm gene column with new info to reuse existing functions
    meta['worm_gene'] = meta['analysis']

    # Print out number of features processed
    feat_filters = [line for line in open(FNAME_FILE) 
                     if line.startswith("#")]
    print ('Features summaries were processed \n {}'.format(
           feat_filters))
    
    # Extract genes in metadata different from control strain and make a list
    # of the total number of straisn
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    
    # Remove strains done from gene list, so we're only analysing the strains
    # we want to
    genes = list(set(genes) - set(strains_done))
    genes.sort()
    strain_numbers = []

    imaging_date_yyyymmdd = meta['date_yyyymmdd']
    imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
    meta['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd                                     
    
    #%% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())
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
        # candidate_gene_colour = STRAIN_cmap[candidate_gene]
        strain_lut = {CONTROL_STRAIN:(0.0, 0.0, 0.0),
                        candidate_gene: (0.6, 0.6, 0.6)}

        if 'all_stim' in ANALYSIS_TYPE:
            print ('all stim plots for {}'.format(candidate_gene))

       #Uses Ida's helper to again select individual strain for analysis
            feat_df, meta_df, idx, gene_list = select_strains([candidate_gene],
                                                              CONTROL_STRAIN,
                                                              feat_df=feat,
                                                              meta_df=meta)
       # Again filter out bad wells, nans and unwanted features
            feat_df_1, meta_df_1, featsets = filter_features(feat_df,
                                                         meta_df)

            strain_lut_old, stim_lut, feat_lut = make_colormaps(gene_list,
                                                            featlist=featsets['all'],
                                                            idx=idx,
                                                            candidate_gene=[candidate_gene],
                                                            )

            #%% Impute nan's and calculate Z scores of features for strains
            feat_nonan = impute_nan_inf(feat_df)

            featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], 
                                                   axis=0),
                                 columns=featsets['all'],
                                 index=feat_nonan.index)

            assert featZ.isna().sum().sum() == 0
            (saveto / 'boxplots').mkdir(exist_ok=True)

            if do_stats:
                    if which_stat_test == 'permutation_ttest':
                        _, unc_pvals, unc_reject = univariate_tests(
                            feat_nonan, y=meta_df['worm_gene'],
                            control='N2',
                            test='t-test',
                            comparison_type='binary_each_group',
                            multitest_correction=None,
                            n_permutation_test=100000,
                            perm_blocks=meta_df['imaging_date_yyyymmdd'],
                            )
                        reject, pvals = _multitest_correct(
                            unc_pvals, 'fdr_by', 0.05)
                        unc_pvals = unc_pvals.T
                        pvals = pvals.T
                        reject = reject.T

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
                    
            # If already calculated, call stats .csv file
            else:
                    bhP_values = pd.read_csv(saveto / '{}_stats.csv'.format(candidate_gene),
                                             index_col=False)
                    bhP_values.rename(mapper={0:'p<0.05'},
                                      inplace=True)
            #%% Import features to be plotted from a .txt file containing the 
            # 3 core feats and plot swarm/boxplots for these
            feat_to_plot_fname = list(feats_plot.rglob('unc-80_feats_to_plot.txt'))[0]
            selected_feats = []
            with open(feat_to_plot_fname, 'r') as fid:
                    for l in fid.readlines():
                        selected_feats.append(l.rstrip().strip(','))

            all_stim_selected_feats=[]
            for s in selected_feats:
                    all_stim_selected_feats.extend([f for f in featsets['all'] if '_'.join(s.split('_')[:-1])=='_'.join(f.split('_')[:-1])])

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
                    plt.tight_layout()
                    plt.savefig(saveto / 'boxplots' / '{}_boxplot.png'.format(f),
                                bbox_inches='tight',
                                dpi=200)
            plt.close('all')
