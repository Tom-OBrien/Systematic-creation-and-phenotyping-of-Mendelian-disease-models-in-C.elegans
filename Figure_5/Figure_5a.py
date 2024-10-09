#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:36:02 2023

This script plots a 3D map of all compounds frmo the initial drug library 
screen vs N2 and unc-80. N2 is plotted as a blue star ('target') and utreated
unc-80 is plotted as a red star. 
All the hits are coloured in blue, with the non-hits in grey.
The hits defined in the 'initial_hits hits_full_name' file were selected by
plotting the 3 core features, and mannually inspecting the plots to identify
drugs that moved unc-80 towards N2 in phenospace.
- The script for doing this is also withint this repository.

@author: tobrien
"""

import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.read_data.hydra_metadata import (align_bluelight_conditions,
                                                   read_hydra_metadata)
sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/Code/Helper_Functions_and_Scripts')
from helper import  filter_features
from initial_hits import hits_full_name
#%% Set paths to the data
FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/InitialScreen/features.csv') 
FNAME_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/InitialScreen/filenames.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/InitialScreen/metadata.csv')
# Set where to save the data
figures_dir = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/test')

# Decide if we're going to analyse the wells marked to be bad
filter_wells=True
# Choose only the handpicked hits
refined_hits_only=False
# Do analysis on handpicked feature set only?
selected_feats_only=True
# Remove unc-80 or N2 from analysis?
remove_unc80=False
remove_N2=False
# Strain with the phenotype we're aiming for
TARGET_STRAIN = 'N2'
# Control strain + DMSO only
CONTROL_STRAIN = 'unc-80_DMSO'
controls = [TARGET_STRAIN, CONTROL_STRAIN]
# Select the core behavioural features to be plotted
selected_features = [
                    'speed_neck_90th_bluelight',
                    'motion_mode_paused_fraction_bluelight',
                    'curvature_mean_head_norm_abs_50th_bluelight',
                    ]
# %% Set plotting style, import and filter data
if __name__ == '__main__':
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

    # Create an analysis column containing the worm gene and drug name, then
    # rename controls as global variables defined above
    meta['analysis'] = meta['worm_gene'] + '_' + meta['drug_type']
    meta['analysis'].replace({'unc_80_DMSO':CONTROL_STRAIN,
                              'N2_DMSO':TARGET_STRAIN},
                             inplace=True)
        
    # To make drug names easier to read, I'm simply stripping the first few characters
    meta['analysis'] = meta['analysis'].map(lambda x: x.lstrip('unc-80_'))
    # The method above renames unc-80 to 'DMSO' so just rename this here
    meta['analysis'].replace({'DMSO':CONTROL_STRAIN},
                            inplace=True)
        
    # Select for inital hits that rescued hand picked feature phenotype, only
    # calculating stats on these is a lot faster
    if refined_hits_only==True:
        mask = meta['analysis'].isin(hits_full_name)
        meta_not_hits = meta[~mask]
        feat_not_hits = feat[~mask]
        meta = meta[mask]
        feat = feat[mask]
        figures_dir = figures_dir / 'Hits_only'
        meta['worm_gene'] = meta['analysis']
        meta_not_hits['worm_gene'] = meta_not_hits['analysis']
    else:
        figures_dir = figures_dir / 'all_compounds_3D_drug_plot'
        
    # The following if statements allow you to remove N2 or unc-80 from plots
    if remove_unc80==True:
        mask = meta['analysis'].isin(['unc-80_DMSO'])
        meta = meta[~mask]
        feat = feat[~mask]
        figures_dir = figures_dir / 'unc-80_removed'
        
    if remove_N2==True:
        mask = meta['analysis'].isin(['N2'])
        meta = meta[~mask]
        feat = feat[~mask]
        figures_dir = figures_dir / 'N2_removed'
  
    #%% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())
    meta = meta.loc[feat.index]
    
    feat, meta, featsets = filter_features(feat,
                                           meta)
    
    # Select hand picked features only and save to new directory
    if selected_feats_only == True:
        feat = feat.drop(columns=[col for col in feat if col 
                                  not in selected_features])

            
    # %% Now I concat everything into one datafram
    data =  pd.concat([feat,
                         meta.loc[:,'analysis']],
                         axis=1)
    
    # To enable fine control of plots, I separate the controls out into new dfs
    N2_data = data.loc[data['analysis'] == 'N2']
    unc_data = data.loc[data['analysis'] == 'unc-80_DMSO']
    
    # Then I make a DF containing all the compound data
    controls = ['N2', 'unc-80_DMSO']
    mask = data['analysis'].isin(controls)
    data = data[~mask]
    
    # To colour the hits/non-hits differently I create 2 new DFs
    mask = data['analysis'].isin(hits_full_name)
    non_hits = data[~mask]
    hits = data[mask]
    
    non_hits= non_hits.groupby('analysis').agg([('mean','mean')])
    z_non=non_hits['curvature_mean_head_norm_abs_50th_bluelight', 'mean']
    x_non=non_hits['motion_mode_paused_fraction_bluelight', 'mean']
    y_non=non_hits['speed_neck_90th_bluelight', 'mean']

    hits=hits.groupby('analysis').agg([('mean','mean')])
    z=hits['curvature_mean_head_norm_abs_50th_bluelight', 'mean']
    x=hits['motion_mode_paused_fraction_bluelight', 'mean']
    y=hits['speed_neck_90th_bluelight', 'mean']
    
    # I now group everything and calculate the mean feature value
    N2_data = N2_data.groupby('analysis').agg([('mean', 'mean')])
    unc_data = unc_data.groupby('analysis').agg([('mean', 'mean')])
    
    # Now we can set all the coordinates for the points 
    # First the drug data

    # Now N2/target data
    z_N2=N2_data['curvature_mean_head_norm_abs_50th_bluelight', 'mean']
    x_N2=N2_data['motion_mode_paused_fraction_bluelight', 'mean']
    y_N2=N2_data['speed_neck_90th_bluelight', 'mean']
    # Now unc-80/control data
    z_unc=unc_data['curvature_mean_head_norm_abs_50th_bluelight', 'mean']
    x_unc=unc_data['motion_mode_paused_fraction_bluelight', 'mean']
    y_unc=unc_data['speed_neck_90th_bluelight', 'mean']   
    

    # Create figure axis
    fig = plt.figure(figsize = (20, 20))
    ax = plt.axes(projection ="3d")
    
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
    
    ax.scatter3D(x_unc, y_unc, z_unc, color = "red", marker='*', s=2500, alpha=0.5)
    ax.scatter3D(x, y, z, s=450, color='cornflowerblue', alpha=0.8)
    ax.scatter3D(x_non, y_non, z_non, s=300, color='grey', alpha=0.2)
    ax.scatter3D(x_N2, y_N2, z_N2, color = "blue", marker='*', s=2500)
    # Disable automatic rotation of z-axis label
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel('Curvature ' '(rads ' '\u03bcm' '$^{-1}$' ')', rotation=90,
                  fontsize=26, labelpad=25)
    ax.set_xlabel('Fraction of paused worms', fontsize=26, labelpad=30)
    ax.set_ylabel('Speed ' '(\u03bcm'' s''$^{-1}$'')', fontsize=26, labelpad=25)   
    
    ax.xaxis.set_tick_params(labelsize=24)    
    ax.yaxis.set_tick_params(labelsize=24)    
    ax.zaxis.set_tick_params(labelsize=24)    
    
    ax.view_init(25, 25)
    # plt.show()
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / 'initial_screen_3D_drug_plot.svg', dpi=300)    