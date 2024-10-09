#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:02:53 2021

Script for making Figure 2 panenel figures, this:
    - Makes PCA plots
    - Makes clustermaps (static and interactive)
    - Plots pre-calculated stats of mutants 
        (Calculated with 'Individual_strain_phenotyping_stats_calculation.py')
    - Plots the example boxplots shown in figure 2

@author: tobrien
"""
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot
import sys
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy import stats
from sklearn.decomposition import PCA
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.preprocessing.filter_data import (filter_nan_inf,
                                                    select_feat_set)

sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/Code/Helper_Functions_and_Scripts')
from helper import (filter_features,
                    make_colormaps,
                    select_strains,
                    STIMULI_ORDER,
                    CUSTOM_STYLE,
                    plot_colormap,
                    plot_cmap_text,
                    feature_box_plots)
from strain_cmap import full_strain_cmap as STRAIN_cmap

#%% The data was collected in two sets, therefore we load them both
#  Data 1
Data1_FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_1/Data1_FeatureMatrix.csv') 
Data1_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_1/Data1_metadata.csv')
# Data 2
Data2_FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_2/Data2_FeatureMatrix.csv')
Data2_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/DataSet_2/Data2_metadata.csv')

# Path to feature orders and sets with stimuli conditions
FEATURE_DIR = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/FeatureSets')
# Path to stats of all strains, calculated using 'Full phenotyping' script
STRAIN_STATS_DIR = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/Strain_Phenotypes_and_Statistics/All_individual_strain_statistics')
# Set the save path
saveto = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/test')
saveto.mkdir(exist_ok=True)

# Choose what figures to plots
plot_PCA = True
plot_Clustermaps = True
plot_interactive_clustermap = True
plot_stats = True
plot_boxes = True

# Choose whether to only examine Tipersy 256 Featureset for eith plot
# In the paper the PCA shows the entire behavioural featureset
PCA_Tierpsy256 = False
# Whereas the clustermap is only for Tierpsy256
Cluster_Tierps256 = True

# Set the Control Strain
CONTROL_STRAIN = 'N2'

   #%% Import the data and filter using Tierpsy functions
if __name__ == '__main__':
 
    Data1_featMat = pd.read_csv(Data1_FEAT_FILE, index_col=False)
    Data1_metadata = pd.read_csv(Data1_METADATA_FILE, index_col=False)
    
    # Optional: Select only Tierpsy256 set
    if PCA_Tierpsy256==True:
        Data1_featMat = select_feat_set(features=Data1_featMat, 
                                   tierpsy_set_name='tierpsy_256', 
                                   append_bluelight=True)
    
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
    # Optional: Select only Tierpsy256 set
    if PCA_Tierpsy256==True:
        Data2_featMat = select_feat_set(features=Data2_featMat, 
                               tierpsy_set_name='tierpsy_256', 
                               append_bluelight=True)
        
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
    
    #%% Set style for all figures
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    # Find all the unique genes within the metadata
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    genes.sort()
    # select only neuro disease models
    meta = meta.query('@genes in worm_gene or @CONTROL_STRAIN in worm_gene')
    feat = feat.loc[meta.index,:]
    
    # Make colour maps for strains and stimuli condition
    strain_lut, stim_lut, feat_lut = make_colormaps(genes,
                                                    featlist=featsets['all']
                                                    )

    plot_cmap_text(strain_lut, 70)
    plt.savefig(saveto / 'strain_cmap.png', bbox_inches="tight", dpi=300)
    plot_colormap(stim_lut, orientation='horizontal')
    plt.savefig(saveto / 'stim_cmap.png', bbox_inches="tight", dpi=300)
    plot_cmap_text(stim_lut)
    plt.savefig(saveto / 'stim_cmap_text.png', bbox_inches="tight", dpi=300)
    plt.close('all')
    
    # No we impute nan's using Tierpsy Tools
    feat_nonan = impute_nan_inf(feat_df)
    #  Calculate Z-score using inbuilt stats package
    featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], axis=0),
                         columns=featsets['all'],
                         index=feat_nonan.index)
    assert featZ.isna().sum().sum() == 0

    #%% PCA plots by strain - prestim, bluelight and poststim separately
    # do PCA on entire space and plot worms as they travel through
     
    if plot_PCA:
        # Make long form feature matrix
        long_featmat = []
        for stim,fset in featsets.items():
            if stim != 'all':
                _featmat = pd.DataFrame(data=feat.loc[:,fset].values,
                                        columns=['_'.join(s.split('_')[:-1])
                                                    for s in fset],
                                        index=feat.index)
                _featmat['bluelight'] = stim
                _featmat = pd.concat([_featmat,
                                      meta.loc[:,'worm_gene']],
                                      axis=1)
                long_featmat.append(_featmat)
        long_featmat = pd.concat(long_featmat,
                                  axis=0)
        long_featmat.reset_index(drop=True,
                                  inplace=True)
        
        full_fset = list(set(long_featmat.columns) - set(['worm_gene', 'bluelight']))
        long_feat_nonan = impute_nan_inf(long_featmat[full_fset])
    
        long_meta = long_featmat[['worm_gene', 'bluelight']]
        long_featmatZ = pd.DataFrame(data=stats.zscore(long_feat_nonan[full_fset], axis=0),
                                      columns=full_fset,
                                      index=long_feat_nonan.index)
        
        assert long_featmatZ.isna().sum().sum() == 0
          #%% Generate PCAs
        pca = PCA()
        X2=pca.fit_transform(long_featmatZ.loc[:,full_fset])
    
        # Explain PC variance using cumulative variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        thresh = cumvar <= 0.95 #set 95% variance threshold
        cut_off = int(np.argwhere(thresh)[-1])
    
        # #Plot as figure
        plt.figure()
        plt.plot(range(0, len(cumvar)), cumvar*100)
        plt.plot([cut_off,cut_off], [0, 100], 'k')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('variance explained')
        plt.tight_layout()
        plt.savefig(saveto / 'long_df_variance_explained.png', dpi =300)
        
        # #now put the 1:cut_off PCs into a dataframe
        PCname = ['PC_%d' %(p+1) for p in range(0,cut_off+1)]
        PC_df = pd.DataFrame(data=X2[:,:cut_off+1],
                              columns=PCname,
                              index=long_featmatZ.index)
    
        PC_plotting = pd.concat([PC_df,
                                  long_meta[['worm_gene',
                                                'bluelight']]],
                                  axis=1)
        
        # groupby worm gene to see the trajectory through PC space
        PC_plotting_grouped = PC_plotting.groupby(['worm_gene',
                                                    'bluelight']).mean().reset_index()
        PC_plotting_grouped['stimuli_order'] = PC_plotting_grouped['bluelight'].map(STIMULI_ORDER)
        PC_plotting_grouped.sort_values(by=['worm_gene',
                                            'stimuli_order'],
                                        ascending=True,
                                        inplace=True)
        
        
        # Calculate standard error of mean of PC matrix computed above
        PC_plotting_sd = PC_plotting.groupby(['worm_gene',
                                              'bluelight']).sem().reset_index()
        # Map to stimuli order of PC_Grouped dataframe
        PC_plotting_sd['stimuli_order'] = PC_plotting_sd['bluelight'].map(STIMULI_ORDER)
        
    #%% Make PC plots of all strains
        save_PCA = saveto / 'PCA_plots'
        save_PCA.mkdir(exist_ok=True)

        plt.figure(figsize = [14,12])    
        s=sns.scatterplot(x='PC_1',
                        y='PC_2',
                        data=PC_plotting_grouped,
                        hue='worm_gene',
                        style='bluelight',
                        style_order=STIMULI_ORDER.keys(),
                        hue_order=strain_lut.keys(),
                        palette=strain_lut,
                        linewidth=0,
                        s=70)
        s.errorbar(
                    x=PC_plotting_grouped['PC_1'],
                    y=PC_plotting_grouped['PC_2'],
                    xerr=PC_plotting_sd['PC_1'], 
                    yerr=PC_plotting_sd['PC_2'],
                    fmt='.',
                    alpha=0.2,
                    )
        ll=sns.lineplot(x='PC_1',
                    y='PC_2',
                    data=PC_plotting_grouped,
                    hue='worm_gene',
                    hue_order=strain_lut.keys(),
                    palette=strain_lut,
                    alpha=0.8,
                    legend=False,
                    sort=False)    
        plt.autoscale(enable=True, axis='both')
        # plt.axis('equal')
        plt.legend(loc='right', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5), fontsize='large')
        plt.xlabel('PC_1 ({}%)'.format(np.round(pca.explained_variance_ratio_[2]*100,2)))
        plt.ylabel('PC_2 ({}%)'.format(np.round(pca.explained_variance_ratio_[3]*100,2)))                                 
        plt.tight_layout()
        plt.savefig(save_PCA / 'PC1PC2_trajectory_space.png', dpi=400)
          
        #%% Make a PC plot of  strains that show a dfferent response to bluelight
        # I.e. bbs-1, bbs-2, unc-43 and  unc-80
        # Make change to the colourmap to select the above strains only
        PC_cmap =  {
                        'N2': ('royalblue'), 
                        'add-1': ('lightgrey'), 
                        'avr-14': ('lightgrey'), 
                        'bbs-1': ('red'),
                        'bbs-2': ('red'),
                        'cat-2': ('lightgrey'), 
                        'cat-4': ('black'), 
                        'dys-1': ('black'), 
                        'figo-1': ('black'), 
                        'glc-2': ('lightgrey'), 
                        'glr-1': ('red'), 
                        'glr-4': ('lightgrey'), 
                        'gpb-2': ('black'), 
                        'kcc-2': ('lightgrey'), 
                        'mpz-1': ('lightgrey'), 
                        'nca-2': ('lightgrey'), 
                        'pink-1': ('lightgrey'), 
                        'snf-11': ('lightgrey'), 
                        'snn-1': ('lightgrey'), 
                        'tmem-231': ('lightgrey'), 
                        'tub-1': ('lightgrey'), 
                        'unc-25': ('black'), 
                        'unc-43': ('red'),
                        'unc-49': ('black'), 
                        'unc-77': ('lightgrey'), 
                        'unc-80': ('red'),
                        }
        
        # Plot PC1 and PC2
        plt.figure(figsize = [14,12])
        # sns.set_style('ticks')
        sp=sns.scatterplot(x='PC_1',
                        y='PC_2',
                        data=PC_plotting_grouped,
                        hue='worm_gene',
                        style='bluelight',
                        style_order=STIMULI_ORDER.keys(),
                        hue_order=PC_cmap.keys(),
                        palette=PC_cmap,
                        linewidth=0,
                        s=70,
                        )
        
        sp.errorbar(
                    x=PC_plotting_grouped['PC_1'],
                    y=PC_plotting_grouped['PC_2'],
                    xerr=PC_plotting_sd['PC_1'], 
                    yerr=PC_plotting_sd['PC_2'],
                    fmt='.',
                    alpha=0.2,
                    )    
        
        l=sns.lineplot(x='PC_1',
                        y='PC_2',
                        data=PC_plotting_grouped,
                        hue='worm_gene',
                        hue_order=PC_cmap.keys(),
                        palette=PC_cmap,
                        alpha=0.8,
                        legend=False,
                        sort=False,)    
            
        plt.autoscale(enable=True, axis='both')
        # plt.legend(loc='right', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5), fontsize='large')
        plt.legend('')
        plt.xlabel('PC_1 ({}%)'.format(np.round(pca.explained_variance_ratio_[0]*100,2)), size=32)
        plt.ylabel('PC_2 ({}%)'.format(np.round(pca.explained_variance_ratio_[1]*100,2)), size=32)              
        plt.xticks(size=30)             
        plt.yticks(size=30)          
        plt.tight_layout()
    
        plt.savefig(save_PCA / 'different_bluelight_response_PC1PC2_trajectory_space.png', dpi=400)
        plt.close('all')
        
#%% Now we plot the clustermap of data from Z-normalised features
    if plot_Clustermaps:
        if Cluster_Tierps256==True:
            FEATURE_DIR = FEATURE_DIR / 'Tierpsy256_FeatureSets'
        else:
            FEATURE_DIR = FEATURE_DIR / 'Entire_FeatureSets'
        
        FEATURES = FEATURE_DIR / 'Z-normalised_features.csv'
        FEATSETS = FEATURE_DIR / 'featsets.npy'
        feat_lut = FEATURE_DIR / 'feat_lut.npy'
                
        
        
        featZ_grouped = pd.read_csv(FEATURES, index_col='worm_gene')
        featsets = np.load(FEATSETS,allow_pickle='TRUE').item()
        feat_lut = np.load(feat_lut,allow_pickle='True').item()
        
        save_cluster = saveto / 'Clustermaps'
        save_cluster.mkdir(exist_ok=True)
        
        # %% Make a colour map and order for fig based on z-normalised features
        strain_lut = STRAIN_cmap
        cluster_group = featZ_grouped.reset_index()
        row_colors = cluster_group['worm_gene'].map(strain_lut)
        
        # make clustermaps
        clustered_features = {}
        sns.set(font_scale=0.8)
        for stim, fset in featsets.items():
                col_colors = featZ_grouped[fset].columns.map(feat_lut)
                plt.figure(figsize=[7.5,5])
                cg = sns.clustermap(featZ_grouped[fset],
                                row_colors=row_colors,
                                col_colors=col_colors,
                                vmin=-1.5,
                                vmax=1.5,
                                )
                cg.ax_heatmap.axes.set_xticklabels([])
                
                cg.savefig(Path(save_cluster) / '{}_clustermap.png'.format(stim), dpi=300)
            
        plt.show()
        plt.close('all')
        
        # %% Now plot interactive clustermap as a static.html
        if plot_interactive_clustermap:
                    cg = sns.clustermap(featZ_grouped[featsets['all']], 
                                                    vmin=-2,
                                                    vmax=2
                                                    # ,metric ='cosine'
                                                    )
                    plt.show()
                    plt.close('all')
                
                    # get order of features and bacteria strain in clustermap
                    row_order = cg.dendrogram_row.reordered_ind
                    col_order = cg.dendrogram_col.reordered_ind     
                
                    # re-order df to match clustering
                    clustered_df_final = featZ_grouped.loc[featZ_grouped.index[row_order], featZ_grouped.columns[col_order]]
        
                    # Define your heatmap
                    intheatmap = (
                        go.Heatmap(x=clustered_df_final.columns, 
                                    y=clustered_df_final.index, 
                                    z=clustered_df_final.values,  
                                    colorscale='Inferno', # Try RdBu or something
                                    zmin=-2,
                                    zmax=2,
                                    showscale=True)
                    )
                    
                    intfig_cl = go.Figure(data=intheatmap)
                    intfig_cl.update_xaxes(showticklabels=False)  
                    intfig_cl.update_yaxes(showticklabels=False, autorange="reversed") 
                    
                    # Define your layout, adjusting colorbar size
                    intfig_cl.update_layout({
                        'width': 1200,
                        'height': 550,
                        'margin': dict(l=0,r=0,b=0,t=0),
                        'showlegend': True,
                        'hovermode': 'closest',
                    })
                    
                    plot(intfig_cl, filename = str(save_cluster / "InteractiveClustermap_cosine.html"),
                          config={"displaylogo": False,
                                  "displayModeBar": True,
                                  "scale":10},
                          auto_open=False 
                )
                
# %% Now plot lineplot and heatmap of stats calculated prior
    if plot_stats ==True:
        save_stats = saveto / 'StatPlots'
        save_stats.mkdir(exist_ok=True)
        strain_stats = [s for s in (STRAIN_STATS_DIR).glob('**/*_stats.csv')]
        print(('Collating pValues for {} worm strains').format(len(strain_stats)))
        
        # Combine all strain stats into one dataframe and reorder columns so worm
        # gene is the first one (easier to read/double check in variable explorer)
        combined_strains_stats = pd.concat([pd.read_csv(f) for f in strain_stats])
        heat_map_row_colors = combined_strains_stats['worm_gene'].map(strain_lut)
        first_column = combined_strains_stats.pop('worm_gene')
        combined_strains_stats.insert(0, 'worm_gene', first_column )
       
        # Save combined stats as a .csv in save directory:
        # combined_strains_stats.to_csv(saveto / "permutation_combined_strain_stats.csv", index=False)
        
        # Set worm gene as index for dataframe- this removes this comlumn from df
        combined_strains_stats = combined_strains_stats.set_index(['worm_gene'])
        
        # Now count total features in df (total features) and save as variable
        total_feats = len(combined_strains_stats.columns)
        
        # Count nan's in df for each strain/row
        null_feats = combined_strains_stats.isnull().sum(axis=1)
        
        # Compute total number of significant feats for each strain
        sig_feats = total_feats - null_feats
        
        # Save as a dataframe (indexed by worm gene)
        sig_feats = pd.DataFrame(sig_feats)
        
        # Naming column containing number of significant feats
        sig_feats = sig_feats.rename(columns={0: 'Total_Significant_Features'}) 
        
        # Sorting dataframe from most -> fewest significant feats
        sig_feats = sig_feats.sort_values(by='Total_Significant_Features', axis=0, 
                                          ascending=False)
        # Resting index on ordered df for purposes of plotting later on
        sig_feats = sig_feats.reset_index()
    
        # Print a summary of the number of significant features
        print('Total number of features {}'.format(total_feats))
        print(sig_feats)
        
        #%% Make a line plot of total significant features ordered save as heatmap
        sns.set_style('ticks')
        l = sns.lineplot(data=sig_feats, 
                          x='worm_gene', 
                          y='Total_Significant_Features',
                          color='black')
        plt.xticks(rotation=90, fontsize=13)
        plt.yticks(rotation=45, fontsize=14)
        l.set_xlabel(' ', fontsize=18)
        l.set_ylabel('Number of Significant Features', fontsize=16, rotation=90)
        plt.yticks([0 ,1000, 2000, 3000, 4000, 5000, 6000, 7000], fontsize=14)
        # l.invert_yaxis()
        l.axhline(y=0, color='black', linestyle=':', alpha=0.2)
        plt.savefig(save_stats / 'sig_feats_lineplot.png', bbox_inches='tight',
                    dpi=300)
        
        #%% Make heatmap of strains showing number of significant features- should be able to define order (not possible using clustermap)
        
        # To make heatmap easy to interpret I set values to either 1 or 0
        # This means that it can be coloured as black-white for sig/non-sig feats
        combined_strain_stat_copy = combined_strains_stats
        heatmap_stats = combined_strain_stat_copy.fillna(value=1)
        
        # I then copy the indexing from the sig feats line plot
        sig_feats = sig_feats.set_index('worm_gene')
        heatmap_stats = heatmap_stats.reindex(sig_feats.index) 
    
        # Here I set colours for the heatmap 
        hm_colors = ((0.0, 0.0, 0.0), (0.95, 0.95, 0.9))
        hm_cmap = LinearSegmentedColormap.from_list('Custom', 
                                                    hm_colors, 
                                                    len(hm_colors))
        
        plt.subplots(figsize=[7.5,5])
        plt.gca().yaxis.tick_right()
        plt.yticks(fontsize=9)
        # Plot the heatmap
        ax=sns.heatmap(data=heatmap_stats,
                        vmin=0,
                        vmax=0.5,
                        xticklabels=False,
                        yticklabels=True,
                        cbar_kws = dict(use_gridspec=False,location="top"),
                        cmap=hm_cmap
                        )
        # Add in the custom coour bar
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.1,  0.4])
        colorbar.set_ticklabels(['P < 0.05', 'P > 0.05'])
        
        ax.set_ylabel('')
    
        plt.savefig(save_stats / 'ordered_formatted_heatmap.png', 
                    bbox_inches='tight', dpi=300)
        
        #%% This does the same as above, but colours the p-val by significance using
        # a log(10) scale. First I reindex using the same method as above
        combined_strains_stats = combined_strains_stats.reindex(sig_feats.index)
        
        # Then we simply plot, I define the min/max sig values based on strain stats
        plt.subplots(figsize=[7.5,5])
        plt.gca().yaxis.tick_right()
        plt.yticks(fontsize=9)
        ax=sns.heatmap(data=combined_strains_stats,
                        # norm=LogNorm(vmin=1e-03, vmax=1e-01),
                        norm=LogNorm(vmin=1e-03, vmax=5e-02),
                        xticklabels=False,
                        yticklabels=True,
                        cbar_kws = dict(use_gridspec=False,
                                        location="top",
                                        # ticks=[1e-03 ,1e-02, 1e-01],
                                        ticks=[1e-03 ,1e-02, 5e-02],
                                        format='%.e'),
                        )
        
        ax.collections[0].colorbar.set_label("P-value")
    
        plt.savefig(save_stats / 'sig_feat_heatmap_w_colours.png', 
                    bbox_inches='tight', dpi=300)

#%% Choose example features and strains to plot as boxplots later 

    if plot_boxes ==True:
        save_box = saveto / 'Boxplots'
        save_box.mkdir(exist_ok=True)
    
        # Plot first example
        EXAMPLES = {'curvature_head_norm_abs_50th_prestim': ['avr-14',
                                                             'unc-25'],}
        strain_lut = {'N2':'lightgrey',
                      'avr-14':'steelblue',
                      'unc-25':'steelblue'}

        for k,v in EXAMPLES.items():
            examples_feat_df, examples_meta_df, dx, gene_list = select_strains(v,
                                                                                CONTROL_STRAIN,
                                                                                feat_df=feat,
                                                                                meta_df=meta)
        
            # filter features
            examples_feat_df, examples_meta_df, featsets = filter_features(examples_feat_df,
                                                          examples_meta_df)
    
            # Using helper function to make a colour map
            examples_strain_lut = make_colormaps(gene_list,
                                                    featlist=featsets['all'],
                                                    idx=dx,
                                                    candidate_gene=v
                                                    )
            examples_strain_lut = examples_strain_lut[0]
        
            # Use my custom function to plot the boxplots
            feature_box_plots(k,
                              examples_feat_df,
                              examples_meta_df,
                              strain_lut,
                              show_raw_data='date',
                              add_stats=False)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(save_box / '{}_boxplot.pdf'.format(k), 
                        bbox_inches="tight",
                        dpi=400)
            plt.close('all')
            
        # Plot second example
        EXAMPLES = {'motion_mode_backward_duration_50th_bluelight': ['kcc-2',
                                                             'mpz-1'],}
        strain_lut = {'N2':'lightgrey',
                      'kcc-2':'steelblue',
                      'mpz-1':'steelblue'}

        for k,v in EXAMPLES.items():
            examples_feat_df, examples_meta_df, dx, gene_list = select_strains(v,
                                                                                CONTROL_STRAIN,
                                                                                feat_df=feat,
                                                                                meta_df=meta)
        
            # filter features
            examples_feat_df, examples_meta_df, featsets = filter_features(examples_feat_df,
                                                          examples_meta_df)
    
            # Using helper function to make a colour map
            examples_strain_lut = make_colormaps(gene_list,
                                                    featlist=featsets['all'],
                                                    idx=dx,
                                                    candidate_gene=v
                                                    )
            examples_strain_lut = examples_strain_lut[0]
        
            # Use my custom function to plot the boxplots
            feature_box_plots(k,
                              examples_feat_df,
                              examples_meta_df,
                              strain_lut,
                              show_raw_data='date',
                              add_stats=False)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(save_box / '{}_boxplot.pdf'.format(k), 
                        bbox_inches="tight",
                        dpi=400)
            plt.close('all')
    
        # Plot third example
        EXAMPLES = {'speed_midbody_norm_10th_prestim': ['glr-1',
                                                        'gpb-2'],}
        strain_lut = {'N2':'lightgrey',
                      'glr-1':'steelblue',
                      'gpb-2':'steelblue'}

        for k,v in EXAMPLES.items():
            examples_feat_df, examples_meta_df, dx, gene_list = select_strains(v,
                                                                                CONTROL_STRAIN,
                                                                                feat_df=feat,
                                                                                meta_df=meta)
        
            # filter features
            examples_feat_df, examples_meta_df, featsets = filter_features(examples_feat_df,
                                                          examples_meta_df)
    
            # Using helper function to make a colour map
            examples_strain_lut = make_colormaps(gene_list,
                                                    featlist=featsets['all'],
                                                    idx=dx,
                                                    candidate_gene=v
                                                    )
            examples_strain_lut = examples_strain_lut[0]
        
            # Use my custom function to plot the boxplots
            feature_box_plots(k,
                              examples_feat_df,
                              examples_meta_df,
                              strain_lut,
                              show_raw_data='date',
                              add_stats=False)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(save_box / '{}_boxplot.pdf'.format(k), 
                        bbox_inches="tight",
                        dpi=400)
            plt.close('all')
    
        # Plot Fourth example
        EXAMPLES = {'width_tail_base_50th_prestim': ['dys-1',
                                                     'add-1'],}
        strain_lut = {'N2':'lightgrey',
                      'dys-1':'steelblue',
                      'add-1':'steelblue'}

        for k,v in EXAMPLES.items():
            examples_feat_df, examples_meta_df, dx, gene_list = select_strains(v,
                                                                                CONTROL_STRAIN,
                                                                                feat_df=feat,
                                                                                meta_df=meta)
        
            # filter features
            examples_feat_df, examples_meta_df, featsets = filter_features(examples_feat_df,
                                                          examples_meta_df)
    
            # Using helper function to make a colour map
            examples_strain_lut = make_colormaps(gene_list,
                                                    featlist=featsets['all'],
                                                    idx=dx,
                                                    candidate_gene=v
                                                    )
            examples_strain_lut = examples_strain_lut[0]
        
            # Use my custom function to plot the boxplots
            feature_box_plots(k,
                              examples_feat_df,
                              examples_meta_df,
                              strain_lut,
                              show_raw_data='date',
                              add_stats=False)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(save_box / '{}_boxplot.pdf'.format(k), 
                        bbox_inches="tight",
                        dpi=400)
            plt.close('all')
    