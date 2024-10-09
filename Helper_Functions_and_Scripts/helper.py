#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:31:14 2020

Helper functions for reading the disease data and making plots in the paper

@author: ibarlow

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re
import itertools
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as PathEffects
from textwrap import wrap
from statannot import add_stat_annotation
from matplotlib import transforms
from tierpsytools.read_data.hydra_metadata import read_hydra_metadata, align_bluelight_conditions
from tierpsytools.preprocessing.filter_data import drop_ventrally_signed, filter_nan_inf, cap_feat_values, feat_filter_std
# %%
CUSTOM_STYLE = '/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/Code/Helper_Functions_and_Scripts/gene_cards.mplstyle'
plt.style.use(CUSTOM_STYLE)

MODECOLNAMES=['frac_worms_fw', 'frac_worms_st', 'frac_worms_bw']

CONTROL_STRAIN = 'N2'
DATES_TO_DROP = '20200626'
BAD_FEAT_THRESH = 3 # 3 standard deviations away from the mean
BAD_FEAT_FILTER = 0.1 # 10% threshold for removing bad features
BAD_WELL_FILTER = 0.3 # 30% threshold for bad well

STIMULI_ORDER = {'prestim':1,
                 'bluelight':2,
                 'poststim':3}

BLUELIGHT_WINDOW_DICT = {0:[55,'prelight',1],
                        1: [70, 'bluelight',1],
                        2: [80, 'postlight',1],
                        3: [155, 'prelight',2],
                        4: [170, 'bluelight',2],
                        5: [180, 'postlight',2],
                        6: [255, 'prelight',3],
                        7: [270, 'bluelight',3],
                        8: [280, 'postlight',3]}

# %%
def drop_nan_worms(feat, meta, saveto, export_nan_worms=False):

    # remove (and check) nan worms
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'date_yyyymmdd']]
    if export_nan_worms:
        nan_worms.to_csv(saveto / 'nan_worms.csv',
                          index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)

    return feat, meta

# %%
def read_disease_data(feat_file, fname_file, metadata_file, drop_nans=True, export_nan_worms=False,
                      align_blue_light=True):

    feat, meta = read_hydra_metadata(feat_file,
                                     fname_file,
                                     metadata_file)
    meta['date_yyyymmdd'] = pd.to_datetime(meta.date_yyyymmdd,
                                                  format='%Y%m%d').dt.date
        
    if align_blue_light == True:
        feat, meta = align_bluelight_conditions(feat,
                                                meta,
                                                how='inner') 
    if drop_nans:
        feat, meta = drop_nan_worms(feat, meta, saveto=feat_file.parent)

    return feat, meta

# %%
def select_strains(candidate_gene, control_strain, meta_df, feat_df=pd.DataFrame()):
    
    gene_list = [g for g in meta_df.worm_gene.unique() if g != control_strain]
    gene_list.sort()
        
    if len(candidate_gene) <=1:
        idx = [c for c,g in list(enumerate(gene_list)) if  g==candidate_gene]
    # else:
    if control_strain not in candidate_gene:
        idx = [gene_list.index(item) for item in candidate_gene]
    else:
        idx=[]
        
    locs = list(meta_df.query('@candidate_gene in worm_gene').index)
    date_to_select = meta_df.loc[locs]['date_yyyymmdd'].unique()
    N2_locs = list(meta_df.query('@date_to_select in date_yyyymmdd and @control_strain in worm_gene').index)
    locs.extend(N2_locs)

     #Only do analysis on the disease strains
    meta_df = meta_df.loc[locs,:]
    if feat_df.empty:
        return meta_df, idx, gene_list
    else:
        feat_df = feat_df.loc[locs,:]

        return feat_df, meta_df, idx, gene_list

# %%
def filter_features(feat_df, meta_df, dates_to_drop=DATES_TO_DROP):

    imgst_cols = [col for col in meta_df.columns if 'imgstore_name' in col]
    miss = meta_df[imgst_cols].isna().any(axis=1)

    # remove data from dates to exclude
    bad_date = meta_df.date_yyyymmdd == float(dates_to_drop)

    # remove wells annotated as bad

    good_wells_from_gui = meta_df.is_bad_well == False
    feat_df = feat_df.loc[good_wells_from_gui & ~bad_date & ~miss,:]
    meta_df = meta_df.loc[good_wells_from_gui & ~bad_date & ~miss,:]

    # remove features and wells with too many nans and std=0
    feat_df = filter_nan_inf(feat_df,
                             threshold=BAD_FEAT_FILTER,
                             axis=0)
    feat_df = filter_nan_inf(feat_df,
                             threshold=BAD_WELL_FILTER,
                              axis=1)

    feat_df = feat_filter_std(feat_df)
    feat_df = cap_feat_values(feat_df)
    feat_df = drop_ventrally_signed(feat_df)

    meta_df = meta_df.loc[feat_df.index,:]
    # feature sets
     # abs features no longer in tierpsy
    pathcurvature_feats = [x for x in feat_df.columns if 'path_curvature' in x]
    #remove these features
    feat_df = feat_df.drop(columns=pathcurvature_feats)

    featlist = list(feat_df.columns)
    featsets={}
    for stim in STIMULI_ORDER.keys():
        featsets[stim] = [f for f in featlist if stim in f]
    featsets['all'] = featlist

    return feat_df, meta_df, featsets

# %%
def make_colormaps(gene_list, featlist, idx=[], candidate_gene=None, CONTROL_DICT={CONTROL_STRAIN:(0.6, 0.6, 0.6)}):

    cmap = list(np.flip((sns.color_palette('cubehelix',
                                           len(gene_list)*2+6))[3:-4:2]))
    N2_cmap = (0.6, 0.6, 0.6)

    strain_lut = {}
    strain_lut[CONTROL_STRAIN] = CONTROL_DICT[CONTROL_STRAIN]

    if candidate_gene is not None:
        for c,g in enumerate(candidate_gene):
            strain_lut[g] = cmap[idx[c]]
    else:
        cmap.append(N2_cmap)
        gene_list.append(CONTROL_STRAIN)
        strain_lut.update(dict(zip(gene_list,
                              cmap)))

    stim_cmap = sns.color_palette('Pastel1',3)
    stim_lut = dict(zip(STIMULI_ORDER.keys(), stim_cmap))

    if len(featlist)==0:
        return strain_lut, stim_lut

    feat_lut = {f:v for f in featlist for k,v in stim_lut.items() if k in f}
    return strain_lut, stim_lut, feat_lut

# %%
def find_window(fname):
    window_regex = r"(?<=_window_)\d{0,9}"
    window = int(re.search(window_regex, str(fname))[0])
    return window

# %%
def plot_colormap(lut, orientation='vertical'):

    sns.set_style('dark')
    plt.style.use(CUSTOM_STYLE)
    
    if orientation == 'vertical':
        tr = transforms.Affine2D().rotate_deg(90)

        fig, ax = plt.subplots(1,1,
                               figsize=[4,5],
                             )
        ax.imshow([[v for v in lut.values()]],
                   transform=tr + ax.transData)
        ax.axes.set_ylim([-0.5, 0.5+len(lut.keys())-1])
        ax.axes.set_yticks(range(0,len(lut.keys())))
        ax.axes.set_yticklabels(lut.keys())

        ax.axes.set_xlim([0.5, -0.5])
        ax.set_xticklabels([])

    else:
        fig, ax = plt.subplots(1,1,
                               figsize=[5,2],
                             )
        ax.imshow([[v for v in lut.values()]])
        ax.axes.set_xticks(range(0, len(lut), 1))
        ax.axes.set_xticklabels(lut.keys(),
                                rotation=45,
                                fontdict={'fontsize':18,
                                          'weight':'bold'})
        ax.axes.set_yticklabels([])
        fig.tight_layout()

    return ax

# %%
def plot_cmap_text(lut, fsize=60):
    plt.figure(figsize = [5,
                          len(lut)*2.5])

    gs1 = gridspec.GridSpec(len(lut),
                           1)
    gs1.update(wspace=-0.01, hspace=0) 

    for c, (k,v) in enumerate(lut.items()):
        ax1 = plt.subplot(gs1[c])
        ax1.text(y=0.5,
                x=0.5,
                s=k,
                verticalalignment='center',
                horizontalalignment='center',
                fontdict={'fontsize':fsize,
                          'weight':'bold',
                          'color':v,
                          'style':'italic' if k!='N2' else 'normal'},
                path_effects=[PathEffects.withStroke(linewidth=1, foreground='k')])
        ax1.axis("off") 
    return

# %%
def make_barcode(heatmap_df, selected_feats, cm=['inferno', 'inferno', 'Greys', 'Pastel1'], vmin_max = [(-1.5,1.5), (-0.5,0.5), (0,4), (1,3)]):
    sns.set_style('ticks')
    plt.style.use(CUSTOM_STYLE)

    fig_ratios = list(np.ones(heatmap_df.shape[0]))
    fig_ratios = [i*3 if c<(len(fig_ratios)-1) else i for c,i in enumerate(fig_ratios)]
    
    f = plt.figure(figsize= (24,3))
    gs = GridSpec(heatmap_df.shape[0], 1,
                  wspace=0,
                  hspace=0,
                  height_ratios=fig_ratios)
    
    cbar_axes = [f.add_axes([.89, .3, .02, .4]), [],
               f.add_axes([.935, .3, .02, .4]),[]]

    for n, ((ix,r), c, v) in enumerate(zip(heatmap_df.iterrows(), cm, vmin_max)):
        axis = f.add_subplot(gs[n])
        
        if ix != 'stim_type' and n<3:
            sns.heatmap(r.to_frame().transpose().astype(float),
                        yticklabels=[ix],
                        xticklabels=[],
                        ax=axis,
                        cmap=c,
                        cbar=n==0 or n==2,
                        cbar_ax=cbar_axes[n],
                        vmin=v[0],
                        vmax=v[1])
            axis.set_yticklabels(labels=[ix],
                                 rotation=0,
                                  fontsize=20
                                 )
        elif ix != 'stim_type':
            sns.heatmap(r.to_frame().transpose().astype(float),
                        yticklabels=[ix],
                        xticklabels=[],
                        ax=axis,
                        cmap=c,
                        cbar=False,
                        vmin=v[0],
                        vmax=v[1])
            axis.set_yticklabels(labels=[ix],
                                 rotation=0,
                                  fontsize=20
                                 )
        else:
            c = sns.color_palette('Pastel1',3)
            sns.heatmap(r.to_frame().transpose(),
                    yticklabels=[ix],
                    xticklabels=[],
                    ax=axis,
                    cmap=c,
                    cbar=n==0,
                    cbar_ax=None if n else cbar_axes[n],
                    vmin=v[0],
                    vmax=v[1])
            axis.set_yticklabels(labels=[ix],
                                 rotation=0,
                                  fontsize=20
                                 )
        cbar_axes[0].set_yticklabels(labels=cbar_axes[0].get_yticklabels())
        cbar_axes[2].set_yticklabels(labels=['>0.05',
                                             np.power(10,-vmin_max[2][1]/2),
                                             np.power(10,-float(vmin_max[2][1]))
                                             ]
                                     ) 
        f.tight_layout(rect=[0, 0, 0.89, 1], w_pad=0.5)

    for sf in selected_feats:
        try:
            axis.text(heatmap_df.columns.get_loc(sf), 1, '*', fontsize=20)
        except KeyError:
            print('{} not in featureset'.format(sf))
    return f
# %%
def make_heatmap_df(fset, featZ, meta, p_vals=None):

    heatmap_df = [pd.concat([featZ,
                        meta],
                       axis=1
                       ).groupby('worm_strain').mean()[fset]]
    if p_vals is not None:
        try:
            heatmap_df.append(-np.log10(p_vals[fset]))
        except TypeError:
            print('p values not logged')
            heatmap_df.append(p_vals[fset])

    _stim = pd.DataFrame(data=[i.split('_')[-1] for i in fset],
                         columns=['stim_type'])
    _stim['stim_type'] = _stim['stim_type'].map(STIMULI_ORDER)
    _stim = _stim.transpose()
    _stim.rename(columns={c:v for c,v in enumerate(fset)}, inplace=True)
    heatmap_df.append(_stim)

    heatmap_df = pd.concat(heatmap_df)
    return heatmap_df
# %%
def feature_box_plots(feature, feat_df, meta_df, strain_lut, show_raw_data=True, bhP_values_df=None, add_stats=True):
    label_format = '{0:.4g}'
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    plt.tight_layout()
    plt.figure(figsize=(6,10))
    ax = sns.boxplot(y=feature,
                x='worm_gene',
                data=pd.concat([feat_df, meta_df],
                               axis=1),
                order=strain_lut.keys(),                
                palette=strain_lut.values(),
                showfliers=False)
    plt.tight_layout()
    if show_raw_data=='date':
        sns.stripplot(y=feature,
                x='worm_gene',
                data=pd.concat([feat_df, meta_df],
                               axis=1),
                order=strain_lut.keys(),
                hue='date_yyyymmdd',
                palette='Greys',
                alpha=0.6)
    ax.set_ylabel(fontsize=22, ylabel=feature)
    ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])
    ax.set_xlabel('')
    ax.set_xticklabels(labels = strain_lut.keys(), rotation=90)
    ax.legend(title='date_yyyy-mm-dd')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if show_raw_data=='drug_conc':
        sns.swarmplot(y=feature,
                x='worm_gene',
                data=pd.concat([feat_df, meta_df],
                               axis=1),
                order=strain_lut.keys(),
                hue='imaging_plate_drug_concentration_uM',
                palette='Greys',
                alpha=0.5)
    ax.set_ylabel(fontsize=22, ylabel=feature)
    ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])
    ax.set_xlabel('')
    ax.set_xticklabels(labels = strain_lut.keys())
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if show_raw_data=='drug_type':
        sns.swarmplot(y=feature,
                x='worm_gene',
                data=pd.concat([feat_df, meta_df],
                               axis=1),
                order=strain_lut.keys(),
                hue='drug_type',
                palette='pastel',
                alpha=0.6)
    ax.set_ylabel(fontsize=22, ylabel=feature)
    ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])
    ax.set_xlabel('')
    ax.set_xticklabels(labels = strain_lut.keys())
    plt.tight_layout()
    if show_raw_data==False:
        pass

    if add_stats:
                
        if bhP_values_df is not None:
            add_stat_annotation(ax,
                                data=pd.concat([feat_df,
                                                meta_df], axis=1),
                                x='worm_gene',
                                y=feature,
                                order=strain_lut.keys(),
                                box_pairs=[strain_lut.keys()],
                                perform_stat_test=False,
                                pvalues=[bhP_values_df[feature].values[0]],
                                test=None,
                                loc='outside',
                                verbose=2,
                                text_annot_custom=['p={:.3E}'.format(round(bhP_values_df[feature].values[0],100))],
                                fontsize=20,
                                )
            plt.tight_layout()

        else:
            add_stat_annotation(ax,
                                data=pd.concat([feat_df,
                                                meta_df], axis=1),
                                x='worm_gene',
                                y=feature,
                                order=strain_lut.keys(),
                                box_pairs=list(itertools.combinations(strain_lut.keys(),2)),
                                perform_stat_test=True,
                                test='t-test_welch',
                                text_format='full',
                                comparisons_correction=None,
                                loc='outside',
                                verbose=2,
                                fontsize=14,
                                line_offset=0.05,
                                line_offset_to_box=0.05
                                )
            plt.tight_layout()

    if len(strain_lut) > 2:
        plt.xticks(rotation=90)
        plt.tight_layout()
    return
# %%
def window_errorbar_plots(feature, feat, meta, cmap_lut, plot_legend=False):
    label_format = '{0:.4g}'
    plt.style.use(CUSTOM_STYLE)

    n_stim = meta.stim_number.unique().shape[0]

    _window_grouped = pd.concat([feat,
                                 meta],
                                axis=1).groupby(['window_sec',
                                                     'worm_gene'])[feature].describe().reset_index()
    fig, ax = plt.subplots(figsize=[(n_stim*2)+2,6])
    for g in meta.worm_gene.unique():
        xs = _window_grouped.query('@g in worm_gene')['window_sec']
        ys = _window_grouped.query('@g in worm_gene')['mean']
        yerr = _window_grouped.query('@g in worm_gene')['mean'] / (_window_grouped.query('@g in worm_gene')['count'])**0.5

        plt.errorbar(xs,
                     ys,
                     yerr,
                     fmt='-o',
                     color=cmap_lut[g],
                     alpha=0.8,
                     linewidth=2,
                     axes=ax,
                     label=g)
        plt.legend('',frameon=False)
    ax.set_ylabel(fontsize=18,
                  ylabel='\n'.join(wrap(feature, 25)))
    ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()],
                       fontsize=16
                       )
    ax.set_xlabel(fontsize=14,
                  xlabel='window')
    ax.set_xticks(ticks=xs)
    ax.set_xticklabels(labels=[x[1] for x in BLUELIGHT_WINDOW_DICT.values() if x[2] in meta.stim_number.unique()],
                        fontsize=12,
                        rotation=45)
    y_min = ax.axes.get_ylim()[0]
    y_max = ax.axes.get_ylim()[1]
    if y_min<y_max:
        rects = (patches.Rectangle((60, y_min), 10, abs(y_min-y_max),
                                   facecolor='tab:blue',
                                   alpha=0.3),
                 patches.Rectangle((160, y_min), 10, abs(y_min-y_max),
                                   facecolor='tab:blue',
                                   alpha=0.3),
                 patches.Rectangle((260, y_min), 10, abs(y_min-y_max),
                                   facecolor='tab:blue',
                                   alpha=0.3))
    [ax.add_patch(r) for r in rects]

    plt.tight_layout()
    return

# %%
def short_plot_stimuli(ax=None, units='s', fps=25,
                 stimulus_start=[60],
                 stimulus_duration=10):
    if ax is None:
        ax = plt.gca()

    if units == 'frames':
        stimulus_start = [s * fps for s in stimulus_start]
        stimulus_duration = stimulus_duration * fps

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    for ss in stimulus_start:
        rect = plt.Rectangle(xy=(ss, ymin),
                             width=stimulus_duration,
                             height=yrange,
                             alpha=0.1,
                             facecolor='tab:blue')
        ax.add_patch(rect)
    return

# %%
def short_plot_frac_by_mode(df,
                      strain_lut,
                      modecolname=MODECOLNAMES[0]):

    plt.figure(figsize=(7.5,5))
    mode_dict = {'frac_worms_fw':'forward',
                 'frac_worms_bw':'backward',
                 'frac_worms_st': 'stationary'}
    
    for strain in list(strain_lut.keys()):
        plt.plot(df[df.worm_gene==strain]['time_s'],
                    df[df.worm_gene==strain][modecolname],
                    color=strain_lut[strain],
                    label=strain, 
                    alpha=0.8)

        lower = df[df.worm_gene==strain][modecolname+'_ci_lower']
        upper = df[df.worm_gene==strain][modecolname+'_ci_upper']
        plt.fill_between(x=df[df.worm_gene==strain]['time_s'],
                             y1=lower.values,
                             y2=upper.values,
                             alpha=0.1,
                             facecolor=strain_lut[strain])

        plt.ylabel('fraction of worms')
        plt.xlabel('time, (s)')
        plt.title(mode_dict[modecolname])
        plt.ylim((0, 1))
        plt.legend(loc='upper right')
        short_plot_stimuli(units='s')
        plt.tight_layout()
    return
# %%
def make_clustermaps(featZ, meta, featsets, strain_lut, feat_lut, group_vars=['worm_gene','date_yyyymmdd'], saveto=None):

    plt.style.use(CUSTOM_STYLE)

    featZ_grouped = pd.concat([featZ,
                               meta],
                              axis=1
                              ).groupby(group_vars).mean()
    featZ_grouped.reset_index(inplace=True)

    row_colors = featZ_grouped['worm_gene'].map(strain_lut)
    clustered_features = {}
    sns.set(font_scale=1.2)
    for stim, fset in featsets.items():
        col_colors = featZ_grouped[fset].columns.map(feat_lut)  
        plt.figure(figsize=[7.5,5])
        cg = sns.clustermap(featZ_grouped[fset],
                        row_colors=row_colors,
                        col_colors=col_colors,
                        vmin=-1,
                        vmax=1,
                        yticklabels=meta['worm_gene'])
        cg.ax_heatmap.axes.set_xticklabels([])
        if saveto!=None:
            cg.savefig(Path(saveto) / '{}_clustermap.png'.format(stim), dpi=300)


        clustered_features[stim] = np.array(featsets[stim])[cg.dendrogram_col.reordered_ind]
        plt.close('all')

    return clustered_features
# %%
def clustered_barcodes(clustered_feats_dict, selected_feats, featZ, meta, p_vals, saveto):

    plt.style.use(CUSTOM_STYLE)
    for stim, fset in clustered_feats_dict.items():

        if stim !='all':
            missing_feats = list(set(fset).symmetric_difference([f for f in featZ.columns if stim in f]))
        else:
            missing_feats = list(set(fset).symmetric_difference(featZ.columns))

        if len(missing_feats)>0:
            for i in missing_feats:
                try:
                    if isinstance(fset, np.ndarray):
                        fset = fset.tolist()
                    fset.remove(i) 
                except ValueError:
                    print('{} not in {} feature set'.format(i, stim))
        heatmap_df = make_heatmap_df(fset, featZ, meta, p_vals)

        f = make_barcode(heatmap_df, selected_feats)
        f.savefig(saveto / '{}_heatmap.png'.format(stim))
    return
# %%
def plot_stimuli(ax=None, units='s', fps=25,
                 stimulus_start=[60, 160, 260],
                 stimulus_duration=10):
    if ax is None:
        ax = plt.gca()

    if units == 'frames':
        stimulus_start = [s * fps for s in stimulus_start]
        stimulus_duration = stimulus_duration * fps

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    for ss in stimulus_start:
        rect = plt.Rectangle(xy=(ss, ymin),
                             width=stimulus_duration,
                             height=yrange,
                             alpha=0.1,
                             facecolor='tab:blue')
        ax.add_patch(rect)
    return

# %%
def plot_frac_by_mode(df,
                      strain_lut,
                      modecolname=MODECOLNAMES[0],
                      smooth=True):
                 
    plt.figure(figsize=(7.5,5))
    mode_dict = {'frac_worms_fw':'forward',
                 'frac_worms_bw':'backward',
                 'frac_worms_st': 'stationary'}
    for strain in list(strain_lut.keys()):
        plt.plot(df[df.worm_gene==strain]['time_s'],
                    df[df.worm_gene==strain][modecolname],
                    color=strain_lut[strain],
                    label=strain,
                    linewidth=2,
                    alpha=0.8)

        lower = df[df.worm_gene==strain][modecolname+'_ci_lower']
        upper = df[df.worm_gene==strain][modecolname+'_ci_upper']
        plt.fill_between(x=df[df.worm_gene==strain]['time_s'],
                             y1=lower.values,
                             y2=upper.values,
                             alpha=0.1,
                             facecolor=strain_lut[strain])

        plt.ylabel('fraction of worms')
        plt.xlabel('time, (s)')
        plt.title(mode_dict[modecolname])
        plt.ylim((0, 1))
        plt.legend(loc='upper right')
        plot_stimuli(units='s')
        plt.tight_layout()
    return