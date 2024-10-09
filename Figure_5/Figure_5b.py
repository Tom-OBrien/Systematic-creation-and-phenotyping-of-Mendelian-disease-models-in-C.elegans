#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:53:08 2023

This script makes the 3D plot shown in Fgiure 5B. The drug names are then 
mannually annotated onto the figure. The points are coloured by high number
of side effects (see paper for details).
-N2 (target strain) is shown as a blue star
-unc-80 (control strain) is shown as a red star

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
from confirmation_hits import hits_full_name

selected_feats_only = True
refined_hits_only = False
filter_wells = True

selected_features = ['speed_neck_90th_bluelight',
                    'motion_mode_paused_fraction_bluelight',
                    'curvature_mean_head_norm_abs_50th_bluelight']
#%% Set paths to the data
FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/ConfirmationScreen/features.csv') 
FNAME_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/ConfirmationScreen/filenames.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/ConfirmationScreen/metadata.csv')
# Set the save path for figures
figures_dir = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/test')
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

    nan_worms.to_csv(
        figures_dir / 'nan_worms.csv', index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)

    if filter_wells==True:
        mask = meta['well_label'].isin([1.0, 3.0])
        
    feat = feat[mask]
    meta = meta[mask]
    
    # Combine information about strain and drug treatment
    meta['analysis'] = meta['worm_gene'] + '+' + meta['drug_type']
    # Rename controls for ease of use
    meta['analysis'].replace({'unc-80+DMSO':'unc-80_DMSO',
                              'N2+DMSO':'N2'},
                             inplace=True)
    
    # Drop water only controls (edges of plate)- we will look at DMSO only
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
    meta['analysis'].replace({'_DMSO':'unc-80'},
                            inplace=True)
    
    meta['worm_gene'] = meta['analysis']
        
    # Select for inital hits that rescued hand picked feature phenotype, only
    # calculating stats on these is a lot faster
    if refined_hits_only==True:
        mask = meta['analysis'].isin(hits_full_name)
        meta_not_hits = meta[~mask]
        feat_not_hits = feat[~mask]
        meta = meta[mask]
        feat = feat[mask]
        figures_dir = figures_dir / 'Hits_only'
        figures_dir.mkdir(exist_ok=True)
    

    # Print out number of features processed
    feat_filters = [line for line in open(FNAME_FILE) 
                     if line.startswith("#")]
    print ('Features summaries were processed \n {}'.format(
           feat_filters))
    
    # Make summary .txt file of feats
    with open(figures_dir / 'feat_filters_applied.txt', 'w+') as fid:
        fid.writelines(feat_filters)

  
    #%% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())
    meta = meta.loc[feat.index]
    
    if refined_hits_only==True:
        feat_not_hits = filter_nan_inf(feat_not_hits, 0.5, axis=1, verbose=True)
        meta_not_hits = meta_not_hits.loc[feat_not_hits.index]
        feat_not_hits = filter_nan_inf(feat_not_hits, 0.05, axis=0, verbose=True)
        feat_not_hits = feat_not_hits.fillna(feat_not_hits.mean())
        meta_not_hits = meta_not_hits.loc[feat_not_hits.index]
        figures_dir = figures_dir / 'refined_hits_only'
        figures_dir.mkdir(exist_ok=True)
    
    # Select hand picked features only 
    if selected_feats_only == True:
        feat = feat.drop(columns=[col for col in feat if col 
                                  not in selected_features])

# %% Now concat everything into one dataframe
    data =  pd.concat([feat,
                         meta.loc[:,'analysis']],
                         axis=1)
    
    # To enable fine control of plots, I separate the controls out into new dfs
    N2_data = data.loc[data['analysis'] == 'N2']
    unc_data = data.loc[data['analysis'] == 'unc-80']
    
    # Then make a DF containing all the compound data
    controls = ['N2', 'unc-80']
    mask = data['analysis'].isin(controls)
    data = data[~mask]
    
    # To look at the high side effect vs other compounds, I create two DFs
    mask = data['analysis'].isin(hits_full_name)
    side_effects_ = data[~mask]
    non_side_effects = data[mask]
    
    side_effects= side_effects_.groupby('analysis').agg([('mean','mean')])
    z_side=side_effects['curvature_mean_head_norm_abs_50th_bluelight', 'mean']
    x_side=side_effects['motion_mode_paused_fraction_bluelight', 'mean']
    y_side=side_effects['speed_neck_90th_bluelight', 'mean']

    non_side_effects=non_side_effects.groupby('analysis').agg([('mean','mean')])
    z=non_side_effects['curvature_mean_head_norm_abs_50th_bluelight', 'mean']
    x=non_side_effects['motion_mode_paused_fraction_bluelight', 'mean']
    y=non_side_effects['speed_neck_90th_bluelight', 'mean']
        
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
    
    ax.scatter3D(x, y, z, s=300, color='green', alpha=0.6)
    ax.scatter3D(x_side, y_side, z_side, s=300, color='orange', alpha=0.6)
    ax.scatter3D(x_N2, y_N2, z_N2, color = "blue", marker='*', s=900)
    ax.scatter3D(x_unc, y_unc, z_unc, color = "red", marker='*', s=900)

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
    plt.savefig(figures_dir / '3D_feat_plot.pdf',
                dpi=300)
