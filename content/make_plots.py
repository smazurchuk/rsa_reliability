#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:43:20 2023

Make methods paper plots

@author: smazurchuk
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc = {'figure.figsize':(12,8)})
sns.set_theme(style="whitegrid")
'''
Set some plot parameters
'''
newRCparams = {
    'font.weight': 'bold',
    'axes.titlesize':'xx-large',
    'axes.titleweight':'bold',
    'axes.labelsize':25,
    'axes.labelweight':'bold',
    'axes.labelpad': 30,
    'xtick.labelsize':20,
    'ytick.labelsize':20,
    'legend.title_fontsize': 'x-large',
    'legend.fontsize':'large',
    'legend.markerscale':2,
    'mathtext.default': 'bf'
    }
plt.rcParams.update(newRCparams)
palette = sns.color_palette(n_colors=6)

# Load in excel data
args   = sys.argv
study  = args[1]
region = args[2]
# study = 'soe'; region = 'ale'

print(f'Study: \t\t{study} \nRegion: \t{region}\n')

# dfs = pd.read_excel(f'{study}_{region}.xlsx',sheet_name=None)
dfs = pd.read_excel(f'excel_results/{study}_{region}.xlsx',sheet_name=None)


'''
This generates the figure with cronbachs resampled and extrapolates
out icc
'''
def sbp(n,r):
    # Spearman-Brown prophecy formula
    out = (n*r) / (1 + (n-1)*r)
    return out
x = np.arange(1,100)
preds = [[sbp(n,dfs['icc2_values'].icc2[k]) for n in x] for k in range(6)]
palette[1], palette[5] = palette[5], palette[1]
tmp = dfs['figure_1_cronbach']
plt.figure()
ax1 = sns.scatterplot(x='numSubj',y='crona',hue='numP', palette=palette,
                      data=tmp.groupby(['numP','numSubj']).mean())
sns.lineplot(x=x,y=preds[0],color=palette[0]); sns.lineplot(x=x,y=preds[1],color=palette[1])
sns.lineplot(x=x,y=preds[2],color=palette[2]); sns.lineplot(x=x,y=preds[3],color=palette[3])
sns.lineplot(x=x,y=preds[4],color=palette[4]); sns.lineplot(x=x,y=preds[5],color=palette[5])
plt.legend(title='Number of Presentations')
plt.xlabel('Number of Participants'); plt.ylabel("Cronbach's Alpha")
#plt.title('Reliability as a function of Stimulus Presentations and Participants')
plt.savefig(f'figures/{study}_{region}_figure_1.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Here is a figure I should have generated before
names = ['Experiential','Distributional','Taxonomic']
for name in names:
    plt.figure()
    ax1 = sns.lineplot(x='numSubj',y=name,hue='numP', palette=palette,
                          data=tmp,style='numP',markers=True)
    plt.legend(title='Number of Presentations')
    plt.xlabel('Number of Participants'); plt.ylabel("Correlation with CREA")
    plt.title(f'Correlation of neural RDM to {name}')
    plt.savefig(f'figures/{study}_{region}_figure_2_{name}.png', dpi=300, bbox_inches='tight')
    # plt.show()


palette[1], palette[5] = palette[5], palette[1]




'''
Create the second figure showing correlation between reliability and 
correlation to different models. I just do everything here
without seaborn
'''
from scipy.optimize import curve_fit 

tmp = dfs['reliability_and_correlation'].groupby('numSubj').mean()
t1  = tmp.sort_values('numSubj').reliability
t2  = tmp.sort_values('numSubj').Experiential
t3  = tmp.sort_values('numSubj').Distributional
t4  = tmp.sort_values('numSubj').Taxonomic

def func(x,a):
    return np.sqrt(a*(x))
a1 = curve_fit(func,t1,t2)[0][0]
R1 = np.corrcoef(t2,func(t1,a1))

a2 = curve_fit(func,t1,t3)[0][0]
R2 = np.corrcoef(t3,func(t1,a2))

a3 = curve_fit(func,t1,t4)[0][0]
R3 = np.corrcoef(t4,func(t1,a3))

plt.figure()
plt.plot(t1,func(t1,a1),color=palette[0])
plt.plot(t1,func(t1,a2),color=palette[1])
plt.plot(t1,func(t1,a3),color=palette[2])
plt.scatter(t1,t2, label='Experiential Model', color=palette[0])
plt.scatter(t1,t3, label='Distributional Model',color=palette[1])
plt.scatter(t1,t4, label='Taxonomic Model',color=palette[2])
plt.xlabel('Reliability'); plt.ylabel('Correlation with Model')
plt.legend(title='Model')
if study  == 'cat':
    plt.annotate(f'$R^2 = {R1[0,1]**2:.5f}$',[.28,.41],fontsize='x-large',color=palette[0])
    plt.annotate(f'$R^2 = {R2[0,1]**2:.5f}$',[.61,.26],fontsize='x-large',color=palette[1])
    plt.annotate(f'$R^2 = {R3[0,1]**2:.5f}$',[.28,.38],fontsize='x-large',color=palette[2])
if study  == 'soe':
    plt.annotate(f'$R^2 = {R1[0,1]**2:.5f}$',[.28,.42],fontsize='x-large',color=palette[0])
    plt.annotate(f'$R^2 = {R2[0,1]**2:.5f}$',[.61,.26],fontsize='x-large',color=palette[1])
    plt.annotate(f'$R^2 = {R3[0,1]**2:.5f}$',[.28,.37],fontsize='x-large',color=palette[2])    
plt.savefig(f'figures/{study}_{region}_figure_3.png', dpi=300, bbox_inches='tight')
# plt.show()


# tmp = dfs['reliability_and_correlation']
# models = tmp.modelName.unique(); 
# corrs  = [tmp[tmp.modelName == i].corr()['reliability']['correlation'] for i in models] 
        
# # Main Figure

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     sns.regplot(x='reliability',y='correlation',
#                 data=tmp[tmp.modelName.isin(['Experiential'])], ci=None)
#     sns.regplot(x='reliability',y='correlation',
#                 data=tmp[tmp.modelName.isin(['Distributional'])], ci=None)
#     sns.regplot(x='reliability',y='correlation',
#                 data=tmp[tmp.modelName.isin(['Taxonomic'])], ci=None)
#     plt.legend(ax.collections[0:2:],['Experiential','Distributional','Taxonomic'],title='Model')
#     plt.title('RSA Correlation and Reliability'); 
#     plt.annotate(f'$R^2 = {corrs[0]**2:.3f}$',[.32,.42],fontsize='x-large',color=palette[0])
#     plt.annotate(f'$R^2 = {corrs[3]**2:.3f}$',[.62,.37],fontsize='x-large',color=palette[1])
#     # plt.show(); #plt.savefig()
    
#     # Supplemental figure
#     plt.figure()
#     for idx,name in enumerate(models[1:]):
#         plt.subplot(2,3,idx+1)
#         sns.regplot(x='reliability',y='correlation',
#                     data=tmp[tmp.modelName.isin([name])], ci=None)
#         plt.xlim([.1,.8]); plt.ylim([.1,.6])
#         plt.title(name); plt.xlabel(None); plt.ylabel(None)
#         plt.annotate(f'$R^2 = {corrs[idx+1]**2:.3f}$',[.2,.5],fontsize='x-large',color=palette[0])
#     # plt.show()
    
    
# if study == 'cat':
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     sns.regplot(x='reliability',y='correlation',
#                 data=tmp[tmp.modelName.isin(['Exp65'])], ci=None)
#     plt.legend(ax.collections[0:2:],['Experiential'],title='Model')
#     plt.title('Hello'); 
#     plt.annotate(f'$R^2 = {corrs[0]**2:.3f}$',[.32,.42],fontsize='x-large',color=palette[0])
#     # plt.show(); #plt.savefig()




'''
Create figure that shows repetition suppression effect
'''
tmp = dfs['repetition_suppresion']
# Main fig
tmp2 = data=tmp[['subj','pres','ses','Experiential']]
fig = plt.figure()
ax = fig.add_subplot(111)
sns.barplot(x='ses',y='Experiential',hue='pres',data=tmp2,alpha=.4)
sns.stripplot(x='ses',y='Experiential',hue='pres',data=tmp2,
              dodge=True,edgecolor='black',linewidth=.5)
plt.legend(ax.collections[:-2:],['1','2'],title='Presentation')
plt.xlabel('Session'); plt.ylabel('Correlation with Experiential Model')
#plt.title('Presentation order and Correlation with Model')
plt.savefig(f'figures/{study}_{region}_figure_4.png', dpi=300, bbox_inches='tight')
# plt.show(); #plt.savefig('figure_3.tiff',dpi=300,bbox_inches='tight')

# Supplement Fig
names = ['Experiential','Distributional','Taxonomic']
plt.figure()
for idx,name in enumerate(names):
    plt.subplot(2,3,idx+1)
    sns.barplot(x='ses',y=name,hue='pres',data=tmp[['subj','pres','ses',name]],alpha=.4)
    sns.stripplot(x='ses',y=name,hue='pres',data=tmp[['subj','pres','ses',name]],
                  dodge=True,edgecolor='black',linewidth=.5)
    plt.legend().remove(); plt.title(name); plt.ylabel(None); plt.ylim([-.02,.08])
    if idx > 0:
        plt.gca().yaxis.set_ticklabels([])
plt.savefig(f'figures/{study}_{region}_figure_5.png', dpi=300, bbox_inches='tight')
# plt.show()


'''
This figure shows different presentation combinations
'''
order  = ['icc_246','icc_123','icc_1234','icc_135','icc_1235','icc_123456']
labels = ['2-4-6','1-2-3','1-2-3-4','1-3-5','1-2-3-5','1-2-3-4-5-6']
# Main figure
plt.figure()
tmp = dfs['model_presentation_combinations']
sns.barplot(x='comb',y='Experiential',color=palette[0], data=tmp,alpha=.4,order=order)
sns.stripplot(x='comb',y='Experiential',color=palette[0],data=tmp, order=order)
plt.xticks(ticks=[0,1,2,3,4,5],labels=labels);
plt.xlabel('Presentation Combination'); plt.ylabel('Correlation with Experiential Model');
plt.savefig(f'figures/{study}_{region}_figure_6.png', dpi=300, bbox_inches='tight')
# plt.show(); #plt.savefig('figure_4.tiff',dpi=300,bbox_inches='tight')

# (tmp[tmp.comb=='icc_135'].Experiential.values - tmp[tmp.comb=='icc_123'].Experiential.values).mean()

# Supplemental with cronbach
plt.figure()
tmp = dfs['cron_fig_combination']; 
sns.barplot(x='comb',y='cron',color=palette[0], data=tmp,alpha=.4,order=order)
plt.xticks(ticks=[0,1,2,3,4,5],labels=labels);
plt.xlabel('Presentation Combination'); plt.ylabel("Cronbach's Alpha");
plt.savefig(f'figures/{study}_{region}_figure_7.png', dpi=300, bbox_inches='tight')
# plt.show(); #plt.savefig('figure_4.tiff',dpi=300,bbox_inches='tight')


'''
Generates noise ceiling figure
'''
tmp = dfs['noise_ceiling']
tmp2 = dfs['noise_ceil_estimate']

fig = plt.figure()
ax = fig.add_subplot(111)
ax = sns.stripplot(x='numSubj',y='value',data=tmp[tmp.measure=='un'],alpha=.3,ax=ax,color=palette[0])
ax = sns.stripplot(x='numSubj',y='value',data=tmp[tmp.measure=='ln'],alpha=.3,ax=ax,color=palette[1])
sns.lineplot(x=np.arange(len(tmp2)),y='eln',data=tmp2, color='black')
plt.legend([plt.gca().collections[0],plt.gca().collections[-2],plt.gca().lines[-1]], 
           ['Upper Noise Ceiling','Lower Noise Ceiling','Analytic Estimate'],labelcolor=palette[:2]+['black']) 
plt.axhline(tmp2.asymptote[0], color='black')
#plt.title('Lower and Upper Noise Ceilings'); 
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55],['5','10','15','20','25','30','35','40','45','50','55','60'])
plt.xlabel('Number of Participants')
plt.ylabel('Correlation with Other Participants'); 
plt.annotate('Expected Asymptotic Value',[0,tmp2.asymptote[0]+.02],fontsize='x-large')
plt.savefig(f'figures/{study}_{region}_figure_8.png', dpi=300, bbox_inches='tight')
# plt.show(); 


# fig = plt.figure()
# tmp = dfs['reliability_tstat'].groupby(['numSubj','rep']).mean()
# plt.scatter(tmp.crona,tmp.d12)
# # plt.show()
# sns.scatterplot(x='crona',y='d12',hue='rep',data=tmp)
