#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:46:55 2023

This script generates excel files used to create the figures
from the neural RDM files

@author: smazurchuk
"""
import sys
import h5py
import utils
import pickle
import multiprocessing
import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr, wilcoxon
from scipy.spatial.distance import pdist, cdist, correlation, squareform

numResamples = 10         # How many resamples at each sample size
numProc      = 12         # How many processors to use?
study        = 'Study_1'  # 'Study_1' or 'Study_2'
region       = 'ALE'      #  select region name
outName      = f'{study}_{region}.xlsx'


# Load data (neural RDMs)
f=open(f'neural_rdms/{study}_{region}_Coef_rdms.pkl','rb'); dsets = pickle.load(f); f.close()
print(f'Study: \t\t{study} \nRegion: \t{region} \nNumSamp: \t{numResamples}')

# Load CREA Ratings (old way)
f = h5py.File('icc_remlOut.hdf5','r')
def pickMaps(maps):
    out = [[k.split('#')[0], idx] for idx,k in enumerate(maps) if 'Tstat' in k and 'RT' not in k]
    out = np.array(out)
    return out[:,0], out[:,1].astype(int)
words,_=pickMaps(f[f'{study}/icc_1/{study}101'].attrs['mapNames'])
aprilTable  = pd.read_excel('data/Ratings_data_April2021.xlsx')
aprilTable['Temperature'] = aprilTable[['Hot','Cold']].max(axis=1)
aprilTable['Texture'] = aprilTable[['Smooth','Rough']].max(axis=1)
aprilTable['Weight'] = aprilTable[['Light','Heavy']].max(axis=1)
catTable = pd.read_csv('data/CREA_SOE_651nouns_mean_ratings.csv')
df = pd.concat([aprilTable, catTable], join='inner', ignore_index=True)
indOrder = []
for word in words:
    indOrder.append( df[df.Word == word].index[0] )
indOrder = np.array(indOrder)
creaRatings = df.iloc[indOrder,1:].to_numpy()
creaRDM = pdist(creaRatings,'cosine').reshape(1,-1)


# Load other models
soe_w2v = pdist(np.load('data/w2v_soe.npy'),'cosine').reshape(1,-1)
cat_w2v = pdist(np.load('data/w2v_cat.npy'),'cosine').reshape(1,-1)

if study == 'Study_1':
    soe_categories  = np.array(utils.loadSemCats(study,wordList=list(words), superC=False))
    soe_supordinate = np.array(utils.loadSemCats(study,wordList=list(words), superC=True))
    soe_categories  = np.array([soe_categories==k for k in np.unique(soe_categories)]).T
    soe_supordinate = np.array([soe_supordinate==k for k in np.unique(soe_supordinate)]).T
    soe_cat_rdm = pdist(np.c_[soe_categories, soe_supordinate]).reshape(1,-1)
if study == 'Study_2':
    cat_categories = np.array(utils.loadSemCats(study,wordList=list(words)))
    cat_categories = np.array([cat_categories==k for k in np.unique((cat_categories))]).T
    cat_cat_rdm = pdist(cat_categories).reshape(1,-1)

# Let's define some helpful functions
def normRows(rdms):
    rdms = rdms - np.nanmean(rdms,axis=1, keepdims=True)
    rdms = rdms / np.nanstd(rdms, axis=1, keepdims=True)
    return rdms

def lowerN(rdms):
    N = len(rdms)
    inds = np.arange(N)
    ln = 1 - np.array([correlation(rdms[i],rdms[inds!=i].mean(0)) for i in range(N)])
    return ln.mean()

def upperN(rdms):
    tmp = rdms.mean(0)
    un = 1 - np.array([correlation(rdms[i],tmp) for i in range(len(rdms))])
    return un.mean()


'''
This generates resampled Cronbach values
'''
print('Starting to compare models ...')
if study == 'soe':
    names = ['Experiential','Distributional','Taxonomic']
    models = np.r_[creaRDM,soe_w2v,soe_cat_rdm]

if study == 'cat':
    names = ['Experiential','Distributional','Taxonomic']
    models = np.r_[creaRDM,cat_w2v,cat_cat_rdm]

# Calculate Cronbach's alpha in random subsamples of the data
use = ['icc_1','icc_12','icc_123','icc_1234','icc_12345','icc_123456']; 
numSubj = []; numP = []; crona = []; experiential = []; distributional = []
taxonomic = []
rng = np.random.default_rng(42)
def corrWmodel(rdms):
    result = 1-cdist(rdms,models,'correlation').squeeze()
    return result
with multiprocessing.Pool(processes=12) as pool:
    for q,k in enumerate(use):
        print(f'##\nWorking on {k}\n##')
        rdms = normRows(dsets[k])
        df_rdms = pd.DataFrame(rdms)
        for i in range(5,38):
            print(f'Working on {i}')
            choices = [rng.choice(len(rdms),i,replace=False) for _ in tqdm(range(numResamples))]
            result1 = [pg.cronbach_alpha(df_rdms.iloc[choice,:].T)[0] for choice in tqdm(choices)]
            result2 = np.array(pool.map(corrWmodel,[np.nanmean(rdms[choice],axis=0,keepdims=True) for choice in choices]))

            numSubj.append(i); numP.append(q+1)
            crona.append(np.mean(result1))
            experiential.append(result2[:,0].mean())
            distributional.append(result2[:,1].mean())
            taxonomic.append(result2[:,2].mean())
        # Add in last point
        numSubj.append(len(rdms))
        numP.append(q+1)
        crona.append(pg.cronbach_alpha(df_rdms.T)[0])
        tmp = corrWmodel(np.nanmean(rdms, axis=0,keepdims=True))
        experiential.append(tmp[0])
        distributional.append(tmp[1])
        taxonomic.append(tmp[2])
df_fig1_crona = pd.DataFrame({'numSubj':numSubj, 'numP':numP, 
                              'crona':crona,'Experiential':experiential,
                              'Distributional':distributional,
                              'Taxonomic':taxonomic})


'''
This generates figure 3. We will use a wide dataframe
'''
rdms = dsets['icc_123456']
rdms = normRows(rdms); df_rdms = pd.DataFrame(rdms)
crona = []; experiential = []; distributional = []; taxonomic = []; numSubj = []
for i in range(5,38):
    print(f'Working on {i}')
    choices = [rng.choice(len(rdms),i,replace=False) for _ in tqdm(range(numResamples))]
    result1 = [pg.cronbach_alpha(data=df_rdms.iloc[choice,:].T)[0] for choice in choices]
    result2 = np.array([1-cdist(rdms[choice].mean(0,keepdims=True),models,'correlation').squeeze() for choice in choices])
    # Let's reshape these
    numSubj.extend(numResamples*[i])
    crona.extend(result1)
    experiential.extend(result2[:,0])
    distributional.extend(result2[:,1])
    taxonomic.extend(result2[:,2])
# Add in last point
tmp = 1-cdist(rdms.mean(0,keepdims=True),models,'correlation')
numSubj.append(len(rdms))
crona.append(pg.cronbach_alpha(data=df_rdms.T)[0])
experiential.append(tmp[0,0])
distributional.append(tmp[0,1])
taxonomic.append(tmp[0,2])
df_rel_and_corr = pd.DataFrame({'reliability':crona, 'numSubj':numSubj,
                                'Experiential':experiential, 'Distributional':distributional,
                                'Taxonomic':taxonomic})
#sns.scatterplot(data=df_fig3, x='reliability',y='correlation', hue='modelName', alpha=.3)


'''
This generates the data for the main figure in the text, along 
with the supplementary figure showing  that the suppresion effect is not 
particular to any model 
'''
use = ['icc_1','icc_2','icc_3','icc_4','icc_5','icc_6']; 
df1 = {'subj':[],'pres':[],'ses':[], 
       'Experiential':[],'Distributional':[],'Taxonomic':[]}
if study == 'soe':
    distributional = soe_w2v; taxonomic = soe_cat_rdm
if study == 'cat':
    distributional = cat_w2v; taxonomic = cat_cat_rdm
p = [1,2,1,2,1,2]
s = [1,1,2,2,3,3]
for idx,key in enumerate(use):
    print(key)
    rdms = dsets[key]
    rdms = normRows(rdms)
    df1['subj'].extend(np.arange(len(rdms)))
    df1['pres'].extend(len(rdms)*[p[idx]])
    df1['ses'].extend(len(rdms)*[s[idx]])
    df1['Experiential'].extend([spearmanr(rdms[i],creaRDM[0])[0] for i in range(len(rdms))])
    df1['Distributional'].extend([spearmanr(rdms[i],distributional[0])[0] for i in range(len(rdms))])
    df1['Taxonomic'].extend([spearmanr(rdms[i],taxonomic[0])[0] for i in range(len(rdms))])
df_repSup = pd.DataFrame(df1)


'''
This generates the resuls for different presentation combinations
'''
use = ['icc_1234','icc_123456','icc_123','icc_135','icc_246','icc_1235']
df1 = {'subj':[], 'comb':[], 'Experiential':[]}
for idx,key in enumerate(use):
    print(key)
    rdms = dsets[key]
    rdms = normRows(rdms)
    df1['subj'].extend(np.arange(len(rdms)))
    df1['comb'].extend(len(rdms)*[key])
    df1['Experiential'].extend([spearmanr(rdms[i],creaRDM[0])[0] for i in range(len(rdms))])
df_fig_comb = pd.DataFrame(df1)

# Same combos, but use cronbach
use = ['icc_1234','icc_123456','icc_123','icc_135','icc_246','icc_1235']
df1 = {'comb':[], 'cron':[]}
for idx,key in enumerate(use):
    print(key)
    rdms = dsets[key]
    rdms = pd.DataFrame(normRows(rdms))
    df1['comb'].append(key)
    df1['cron'].append(pg.cronbach_alpha(rdms.T)[0])
df_fig_cron_comb = pd.DataFrame(df1)


'''
This generates the data for the noise ceiling plot. For plotting convienence,
I've generated it as a tall dataframe

So that the script generating the figures only requires a single excel file, I've also 
calculated the bounds here
'''
rdms = dsets['icc_123456']
rdms = normRows(rdms); 
numSubj = []; measure = []; value=[]; 
with multiprocessing.Pool(processes=12) as pool:
    for i in range(5,38):
        print(f'Working on {i}')
        choices = [rng.choice(len(rdms),i,replace=False) for _ in tqdm(range(numResamples))]
        result1  = np.array(pool.map(lowerN, [rdms[choice] for choice in choices]))
        result2  = np.array(pool.map(upperN, [rdms[choice] for choice in choices]))
        numSubj.extend(numResamples*[i])
        measure.extend(numResamples*['ln'])
        value.extend(result1)
        numSubj.extend(numResamples*[i])
        measure.extend(numResamples*['un'])
        value.extend(result2)
df_noiseCeil = pd.DataFrame({'numSubj':numSubj, 'measure':measure,'value':value})

# Make estimate
def EstimateLN(subjRDMs):
    N = subjRDMs.shape[0]      # Total number of subjects
    A = subjRDMs.var(1).mean() # Mean of variance
    D = subjRDMs.mean(0).var() # Variance of mean
    B = (A-(N*D))/(1-N)     # Estimated Signal Variance
    C = (A-D)/(1-(1/N))     # Estimated Noise Variance
    ln = []                 # Initialize array to save output
    for i, n in enumerate(np.arange(5,61)):
        sigmaY = np.sqrt(B + ((1/(n-1))*C))
        ln.append( B / (np.sqrt(A)*sigmaY) )
    asymptote = B / (np.sqrt(A)*np.sqrt(B))
    return ln, asymptote
eln, asymptote = EstimateLN(rdms)
df_noiseCeil_estimate = pd.DataFrame({'numSubj':np.arange(5,61),'eln':eln,'asymptote':asymptote})


'''
This takes a while, so I don't reccoment running it often
I have written it as a function just so that it can be at least a bit parallel
'''
use = ['icc_1','icc_12','icc_123','icc_1234','icc_12345','icc_123456']; 
# Calculate ICC2 Values
def calcIcc(rdms):
    rdms = normRows(rdms)
    df   = pd.DataFrame(rdms)
    df['Subj'] = np.arange(1,len(rdms)+1)
    tmp         = pd.melt(df, id_vars='Subj')
    icc = pg.intraclass_corr(data=tmp,
                             targets = 'variable',
                             raters  = 'Subj',
                             ratings = 'value',
                             nan_policy = 'omit')
    icc_value = icc[icc.Type == 'ICC2'].ICC.item()
    return icc_value
# Load paralle pool
print('Starting to calculate ICC Value ...')
with multiprocessing.Pool(processes=len(use)) as pool:
    icc2 = pool.map(calcIcc, [dsets[key] for key in use])
df_icc = pd.DataFrame({'name':use,'icc2':icc2})



'''
This section saves out all the different dataframes
generated in this script!
'''

with pd.ExcelWriter(outName) as writer:
    # Figure 1
    df_fig1_crona.to_excel(writer, sheet_name='figure_1_cronbach')
    df_icc.to_excel(writer, sheet_name='icc2_values')
    # Figure 2
    df_rel_and_corr.to_excel(writer, sheet_name='reliability_and_correlation')
    # Figure 3
    df_repSup.to_excel(writer, sheet_name='repetition_suppresion')
    # Figure 4
    df_fig_comb.to_excel(writer, sheet_name='model_presentation_combinations')
    # Figure 5
    df_fig_cron_comb.to_excel(writer, sheet_name='cron_fig_combination')
    # Noise ceiling fig
    df_noiseCeil.to_excel(writer, sheet_name='noise_ceiling')
    df_noiseCeil_estimate.to_excel(writer, sheet_name='noise_ceil_estimate')
    # Reliability and T-stat correlation
    df_rel_diff.to_excel(writer, sheet_name='reliability_tstat')

print('All done! Script ran to completion!')






