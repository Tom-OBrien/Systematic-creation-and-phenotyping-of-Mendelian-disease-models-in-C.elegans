#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:36:11 2022

Script for making scatter plot of gene homology vs genetic homology of strains
and scatter plot of BLAST expected value scores

@author: tobrien
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

gene_info = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/DataSets/percentage_homology_blast_scores_and_number_or_orthology_programs.csv')
gene_info = pd.read_csv(gene_info)
genes = gene_info.sort_values(by=['worm_gene'], ascending=True)
saveto = Path('/Users/tobrien/Documents/Zenodo/Systematic creation and phenotyping of Mendelian disease models in C. elegans/test')

sns.scatterplot(data=genes,
            x='Gene_similarity', 
            y='No_OfPrograms',  
            hue=None, 
            s=100,
            color='black',
            legend=None)
plt.xticks(ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
           labels=['0','10','20','30','40','50','60','70','80','90','100'])
plt.xlabel('Genetic Homology (%)')
plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], 
           labels=['0','1','2','3','4','5','6'])
plt.ylabel('No. Programs Predicting Gene is an Ortholog')
plt.savefig(saveto / 'homology_vs_programs.png', dpi=600)
plt.show()
plt.close('all')

# Add 1 to all blast e-vales

sns.boxplot(x='worm_gene',
            y='Blast_e-value',
            data=genes)
plt.yscale('log')
plt.xlabel('Worm Gene')
plt.xticks(rotation=90)
plt.ylabel('Blast e-value')
