#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:23:22 2023

This script collates the stats calculated using the:
    'Calculate_stats_of_confirmation_screen.py' script. It orders them into a 
    .csv file, that was then used to hardcode the Fig.5C barplot

@author: tobrien
"""
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt 
from tierpsytools.preprocessing.filter_data import select_feat_set

sys.path.insert(0, '/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/Scripts')
from confirmation_hits import hits_full_name

METADATA = Path('/Users/tobrien/Documents/Imperial : MRC/unc-80_repurposing_confirmation_screen/AuxiliaryFiles/wells_updated_metadata.csv')
CONTROL = 'unc-80'
order_plot=True

remove_hits = False

t_perm_test_or_ANOVA = 'perm_ttest'
if t_perm_test_or_ANOVA == 'perm_ttest':
    ROOT_STAT_DIR = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DrugRepurposing/ConfirmationScreen/old_stats')
    saveto = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/test')

# First find the control stats
control_stats = pd.read_csv(ROOT_STAT_DIR / CONTROL /'{}_stats.csv'.format(CONTROL))

# I now make a list of all features
full_feat_set = control_stats.columns.values.tolist()
# Now I make a list of all the features that are NOT significant 
ns_feat_list = control_stats.columns[control_stats.isnull().any()].tolist()
# Now I get a list of the significant features 
sig_feats = [x for x in full_feat_set if x not in ns_feat_list]
# Then assert that we've calculated this correctly
assert(len(sig_feats) + len(ns_feat_list) == len(full_feat_set))

#%% Now we want to look at the Tierpsy 256 featureset only
control_256_feats = select_feat_set(features=control_stats, 
                                    tierpsy_set_name='tierpsy_256', 
                                    append_bluelight=True)

# I now make a list of all 256 features
feat_set_256 = control_256_feats.columns.values.tolist()
# Now I make a list of all the features that are NOT significant 
ns_feat256 = control_256_feats.columns[control_256_feats.isnull().any()].tolist()
# Now I get a list of the significant features 
sig_feats256 = [x for x in feat_set_256 if x not in ns_feat256]
# Then assert that we've calculated this correctly
assert(len(sig_feats256) + len(ns_feat256) == len(feat_set_256))

#%% Now lets find all the compounds in the metadata and make a list
meta = pd.read_csv(METADATA)

# Remove brand name from compounds in metadata
meta['drug_type'].replace({'Abitrexate (Methotrexate)':'Abitrexate',
                          'Clozapine (Clozaril)':'Clozapine',
                          'Iloperidone (Fanapt)':'Iloperidone',
                          'Sulindac (Clinoril)':'Sulindac',
                          'Atorvastatin calcium (Lipitor)':'Atorvastatin calcium',
                          'Mesalamine (Lialda)':'Mesalamine',
                          'Fenofibrate (Tricor, Trilipix)':'Fenofibrate',
                          'Mitotane (Lysodren)':'Mitotane',
                          'Ivabradine HCl (Procoralan)':'Ivabradine HCl',
                          'Daunorubicin HCl (Daunomycin HCl)':'Daunorubicin HCl',
                          'Ciprofloxacin (Cipro)':'Ciprofloxacin'}, 
                          inplace=True)

drugs = [d for d in meta['drug_type'].unique()]
# Now I remove all the unwanted controls from plot
drugs.remove('DMSO')
drugs.remove('no compound')
drugs.remove('water')

if remove_hits==True:
    saveto = saveto / 'hits_only'
    drugs = [x for x in drugs if x in hits_full_name]
    
#%% Now we want to iterate over all the drugs and find changes in sig feats

# Make an empty dataframe to append results in
rescue_vs_side_effects = pd.DataFrame(columns=['drug', 'rescued_feats', 
                                               'side_effects'])

rescue_vs_side_effects_256 = pd.DataFrame(columns=['drug', 'rescued_feats', 
                                               'side_effects'])
for count, d in enumerate(drugs):
        print('Analysing {} {}/{}'.format(d, count+1, len(drugs)))
        candidate_drug = d
        
        # First, locate the drug stat file
        drug_stats = pd.read_csv(ROOT_STAT_DIR/ d / '{}_stats.csv'.format(d))
        # Also select the 256 feature set 
        drug_stats_256 = select_feat_set(features=drug_stats, 
                                         tierpsy_set_name='tierpsy_256', 
                                         append_bluelight=True)
        
        # Now I save a separate dataframe for the unc-80 sig/non-sig stats
        drug_stats_significant = drug_stats.loc[:, sig_feats]
        drug_stats_not_sig = drug_stats.loc[:, ns_feat_list]
        
        # Same again with the 256 set
        drug_256_significant = drug_stats_256.loc[:, sig_feats256]
        drug_256_not_sig = drug_stats_256.loc[:, ns_feat256]

        # Because ns values are saved as nans, we can just count them
        rescued_feats = drug_stats_significant.isna().sum().sum()
        rescued_256 = drug_256_significant.isna().sum().sum()
        
        # To calculate side effects: subtract number of significant feats from
        # total length of unc-80 non-significant feats
        side_effects = drug_stats_not_sig.isna().sum().sum()
        side_effects = len(ns_feat_list) - side_effects
        
        side_effects_256 = drug_256_not_sig.isna().sum().sum()
        side_effects_256 = len(ns_feat256) - side_effects_256
        
        # Save this data as a new dataframe and append to main DF
        data = [[d, rescued_feats, side_effects]]
        df = pd.DataFrame(data, columns=['drug', 'rescued_feats', 'side_effects'])
        rescue_vs_side_effects = pd.concat([rescue_vs_side_effects, df], 
                                           ignore_index=True, sort=False)
        
        data_256 = [[d, rescued_256, side_effects_256]]
        df_256 = pd.DataFrame(data_256, columns=['drug', 'rescued_feats', 'side_effects'])
        rescue_vs_side_effects_256 = pd.concat([rescue_vs_side_effects_256, df_256],
                                               ignore_index=True, sort=False)
        
if order_plot==True:
    rescue_vs_side_effects = rescue_vs_side_effects.sort_values(['side_effects'], ascending=False).reset_index(drop=True)
    rescue_vs_side_effects_256 = rescue_vs_side_effects_256.sort_values(['side_effects'], ascending=False).reset_index(drop=True)

rescue_vs_side_effects.to_csv(
                        saveto/'rescued_feats_vs_side_effects.csv',
                        index=False)
        
rescue_vs_side_effects_256.to_csv(
                        saveto/'rescued_feats_vs_side_effects_TIERPSY256.csv',
                        index=False)
        
#%% First, lets save the data as two separate bar plots

# The easiest way to make a nice plot for the paper, was to hardcode it, here
# the results from the 'side effets.csv' file are used to hardcode the plot
custom_color = ['green','green','green','green','green','green','green','green','green','green','green','green','green','green','green','green','green','green','green','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange']
y=[106,143,157,170,205,206,212,233,313,325,339,342,343,368,496,553,566,573,654,1312,2408,2559,2565,2840,3255,3335,3421,3655,3996,4063]
x=['Liranaftate','Rizatriptan','Ivabradine','Mesalamine','Vinblastine','Sulindac','D-Cycloserine','Rofecoxib','Sulfadoxine','Ofloxacin','Olanzapine','Abitrexate','Carbenicillin','Ciprofloxacin','Moxifloxacin','Atorvastatin','Daunorubicin','Norfloxacin','Idarubicin','Mitotane','Loratadine','Fenofibrate','Amitriptyline','Azatadine','Clozapine','Ziprasidone','Mirtazapine','Detomidine','Iloperidone','Medetomidine',]


plt.bar(x=x,
            height=y,
            color=custom_color)
plt.axvline(x=18.5,linewidth=1, color='r', linestyle='dashed', alpha=0.8)
plt.margins(x=0.01) 
# sns.barplot(x=rescue_vs_side_effects['drug'],
#             y=rescue_vs_side_effects['side_effects'],
#             color=custom_color)
plt.xticks(rotation=90,
           fontsize=10)
plt.xlabel('')
plt.ylabel('Number of Worsened Features')
plt.savefig(saveto/ 'side_effects.pdf',
            # order='descending',
            bbox_inches='tight', dpi=300)
plt.show()
plt.close('all')

