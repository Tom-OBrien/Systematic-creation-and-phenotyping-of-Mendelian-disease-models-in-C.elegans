#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 09:03:24 2022

Script for making combined box & swarm plots of unc-80 reprposing data:
plots handpicked features defined as global variables for each strain or from
an array of 'hits' passed in 'inital_hits.py' 
    -'refined_hits' were selected upon manual inspection of data and used to
    re-run the code to save in separate folder for ease of analysis

@author: tobrien
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.read_data.hydra_metadata import (align_bluelight_conditions,
                                                   read_hydra_metadata)
sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/Code/Helper_Functions_and_Scripts')
from helper import  CUSTOM_STYLE
from initial_hits import hits_full_name
   
#%%

# Set paths to the data
FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/InitialScreen/features.csv') 
FNAME_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/InitialScreen/filenames.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/InitialScreen/metadata.csv')
saveto = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/test')

# Decide if we're going to analyse the wells marked to have percipitated
filter_wells=True
# Decide which compounds to run, hits will be saved in a separate folder
refined_hits_only=True
all_compound_boxplots=True

# Phenotypic features we're looking for rescue on (core unc-80 phenotype)
FEATURES = ['speed_neck_90th_bluelight',
            'motion_mode_paused_fraction_bluelight',
            'curvature_mean_head_norm_abs_50th_bluelight']
# Set features as variables for calling while plotting
f0 = FEATURES[0]
f1 = FEATURES[1]
f2 = FEATURES[2]

# Strain with the phenotype we're aiming for
TARGET = 'N2'
# Control strain + DMSO only
CONTROL = 'unc-80_DMSO'
controls = [TARGET, CONTROL]

#%%
if __name__ == '__main__':
    # Set styles for plots
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    # Read in data and align by bluelight with tierpsy tools functions
    feat, meta = read_hydra_metadata(
        FEAT_FILE,
        FNAME_FILE,
        METADATA_FILE)
    # Now align by blue light using Tierpsy tools
    feat, meta = align_bluelight_conditions(
        feat, meta, how='inner')
    
    # Filter out nan's within specified columns and print .csv of these to check    
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'date_yyyymmdd']]
    nan_worms.to_csv(
        saveto/ 'nan_worms.csv', index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)
    # Drop wells containing neither DMSO or a compound from both dataframes
    mask = meta['drug_type'].isin(['empty'])
    feat = feat[~mask]
    meta = meta[~mask]
    
    # Drop or keep marked with percipitation (annotation of 1 or 3)
    meta['well_label']=meta['well_label'].values.astype(str)
    if filter_wells==True:
        mask = meta['well_label'].isin(['1.0', '3.0'])
        
    feat = feat[mask]
    meta = meta[mask]
    
    # Drop compounds not in FDA approved library- from manual inspection these
    # did not rescue the phenotype we are looking for
    mask = meta['drug_type'].isin(['Procaine_HCl',
                                           'Ambroxol_HCl',
                                           'Carbamazepine'])
    feat = feat[~mask]
    meta = meta[~mask]
    # Merging drug names and worm genes together, then renaming controls
    meta['analysis'] = meta['worm_gene'] + '_' + meta['drug_type']
    meta['analysis'].replace({'unc_80_DMSO':CONTROL,
                              'N2_DMSO':TARGET},
                             inplace=True)

    # To make drug names easier to read, I'm simply stripping the first few characters
    meta['analysis'] = meta['analysis'].map(lambda x: x.lstrip('unc-80_'))
    # The method above renames unc-80 to 'DMSO' so just rename this here
    meta['analysis'].replace({'DMSO':CONTROL},
                            inplace=True)
    # This will select the hits we already 
    if refined_hits_only==True:
        mask = meta['analysis'].isin(hits_full_name)
        meta = meta[mask]
        feat = feat[mask]
        saveto = saveto / 'refined_hits_only'
        saveto.mkdir(exist_ok=True)
    
    # Obtain a list of all unique drugs values and remove controls from it
    unique_values = [d for d in meta.analysis.unique()] 
    drugs = [x for x in unique_values if x not in controls]

    # Set date to nicer format for plotting
    meta['date_yyyymmdd'] = pd.to_datetime(
        meta['date_yyyymmdd'], format='%Y%m%d').dt.date
    # To keep data compatible with old helper functions, copy 'date' column
    # as imaging date column 
    imaging_date_yyyymmdd = meta['date_yyyymmdd']
    imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
    meta['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd

    #% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())

    #%% Use subplots to plot handpicked control vs target vs different comppound
    # using subplot
    
    # Print a counting timer of drugs being analysed 
    if all_compound_boxplots ==True:
        for count, d in enumerate(drugs):
            print('Analysing {} {}/{}'.format(d, count+1, len(drugs)))
            candidate_drug = d
            drug_lut = {TARGET:(0.0, 0.0, 0.0),
                        CONTROL:(1.0, 1.0, 1.0),
                        candidate_drug: (0.6, 0.6, 0.6)}
        
            # Set some layout styles
            label_format = '{0:.4g}'
            plt.style.use(CUSTOM_STYLE)
            sns.set_style('ticks')
            plt.tight_layout()
            
            # Create subplot axis and title for plots
            fig, axes = plt.subplots(1,3, sharey=False, figsize=(24,12))
            fig.suptitle('{}'.format(d), fontsize=30)
            
            # Plot box and swarm plots for first feature
            sns.boxplot(ax=axes[0],
                        y=f0,
                        x='analysis',
                        data=pd.concat([feat, meta],
                                       axis=1),
                        order=drug_lut.keys(),                
                        palette=drug_lut.values(),
                        showfliers=False)
            sns.swarmplot(ax=axes[0],
                    y=f0,
                    x='analysis',
                    data=pd.concat([feat, meta],
                                   axis=1),
                    order=drug_lut.keys(),
                    hue='date_yyyymmdd',
                    palette='Greys',
                    alpha=0.6)
            axes[0].set_ylabel(fontsize=26, ylabel=f0)
            axes[0].set_yticklabels(labels=[label_format.format(x) for x in axes[0].get_yticks()])
            axes[0].set_xlabel('')
            axes[0].set_xticklabels(labels = drug_lut.keys(), rotation=90)
            axes[0].get_legend().remove()
            
            # Plot box and swarm plots for second feature
            sns.boxplot(ax=axes[1],
                        y=f1,
                        x='analysis',
                        data=pd.concat([feat, meta],
                                       axis=1),
                        order=drug_lut.keys(),                
                        palette=drug_lut.values(),
                        showfliers=False)
            sns.swarmplot(ax=axes[1],
                    y=f1,
                    x='analysis',
                    data=pd.concat([feat, meta],
                                   axis=1),
                    order=drug_lut.keys(),
                    hue='date_yyyymmdd',
                    palette='Greys',
                    alpha=0.6)
            axes[1].set_ylabel(fontsize=26, ylabel=f1)
            axes[1].set_yticklabels(labels=[label_format.format(x) for x in axes[1].get_yticks()])
            axes[1].set_xlabel('')
            axes[1].set_xticklabels(labels = drug_lut.keys(), rotation=90)
            axes[1].get_legend().remove()
            
            # Plot box and swarm plots for third feature
            sns.boxplot(ax=axes[2],
                        y=f2,
                        x='analysis',
                        data=pd.concat([feat, meta],
                                       axis=1),
                        order=drug_lut.keys(),                
                        palette=drug_lut.values(),
                        showfliers=False)
            sns.swarmplot(ax=axes[2],
                    y=f2,
                    x='analysis',
                    data=pd.concat([feat, meta],
                                   axis=1),
                    order=drug_lut.keys(),
                    hue='date_yyyymmdd',
                    palette='Greys',
                    alpha=0.6)
            axes[2].set_ylabel(fontsize=26, ylabel=f2)
            axes[2].set_yticklabels(labels=[label_format.format(x) for x in axes[2].get_yticks()])
            axes[2].set_xlabel('')
            axes[2].set_xticklabels(labels = drug_lut.keys(), rotation=90)
            axes[2].legend(title='date_yyyy-mm-dd',
                           fontsize='xx-large')
            
            plt.savefig(saveto / '{}.png'.format(d), bbox_inches='tight')
            plt.close('all')
