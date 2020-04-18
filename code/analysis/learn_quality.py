# Investigating of the relation between translationese and quality on morpho-syntactic features for refined collection of good-bad translations
# initially created as a ipynb Jul22, 2019; converted and updated April 18,2020
# How good are our 45 features for learning hand annotated quality?

import sys
import pandas as pd
import argparse
# import numpy as np
# from numpy import dot
# from numpy.linalg import norm
# from numpy import median
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # from functions import *
#
# # import the functions from the helper scripts
from HTQ_functions import HTQ_get_xy, HTQ_pca_transform, nese_visualizePCA, crossvalidate, HTQ_visualizePCA, HTQ_textsdensity
from HTQ_functions import quantify_diffs # plot_confusion_matrix, run_SVM, cross_validate_dummy
# from HTQ_functions import  , , HTQ_textsdensity
# from HTQ_functions import gridsearch_RF, cross_validate_RF
# from smart_open import open

parser = argparse.ArgumentParser()
parser.add_argument('-testdata', help="Path to tsv with feature vals for good-bad ", required=True) # /home/u2/proj/done/HiT-IT/data/good-bad_refined.tsv
# processed makes sense as a dirname
parser.add_argument('-refdata', default="/home/u2/proj/done/HiT-IT/data/pro385-ref375.tsv")

args = parser.parse_args()


# HTQ-labeled data: graded targets available for newspaper register (April 2020) and the same limited to shared sources only (40 sources)
df = pd.read_csv(args.testdata, delimiter="\t")

todrop = df[df.alang == 'en'].index
df0 = df.drop(todrop)
# print(len(df0))
print('Good:', len(df0.loc[df0.astatus == 'good']))
print('Bad:', len(df0.loc[df0.astatus == 'bad']))

# translationese data: ref-pro
df_ref = pd.read_csv(args.refdata, delimiter="\t")

ref_pro = df_ref.loc[(df_ref['alang'] == 'ru') & ((df_ref['astatus'] == 'ref') | (df_ref['astatus'] == 'pro'))]

print('Non-translations(ref):', len(ref_pro.loc[(ref_pro.alang == 'ru') & (ref_pro.astatus == 'ref')])) 
print('Pro translations:', len(ref_pro.loc[(ref_pro.alang == 'ru') & (ref_pro.astatus == 'pro')]))

# do our features capture differences between classes?

# your X and Y for the selected number of best features and the new df adjusted to the new number of column
Xpro45, Ypro, _ = HTQ_get_xy(ref_pro, class_col='astatus', features=None, scaling=1, select_mode='RFE')
# Xpro45pca, totvar_pro45pca, feats_pro45pca = HTQ_pca_transform(Xpro45, ref_pro, 2, 'Dim1', print_best=15)
# nese_visualizePCA(Xpro45pca, Ypro, totvar_pro45pca, dimx=1, dimy=2, feats=45)

# verify by classification:
crossvalidate(Xpro45,Ypro, algo='SVM', grid=0, cv=10, class_weight = 'balanced') ## try RF with grid search :-)
# print('=========Compare to a dummy classifier baseline=========')
# crossvalidate(Xpro45,Ypro, algo='dummy', grid=0, cv=10, class_weight = 'balanced')

# test whether the translationese effect is detectable in student translations as one class or (below) respecting the annotated quality classes
# build a joint df with ref as the third class
ref = df_ref.loc[(df_ref['alang'] == 'ru') & (df_ref['astatus'] == 'ref')]

ref_qua = pd.concat([ref,df0], axis=0, join='outer', sort=False)
# print(ref_qua.shape)

# create transl class variable to treat good/bad as one class
ref_qua0 = ref_qua.copy()
ref_qua0.loc[(ref_qua0.astatus == 'good') | (ref_qua0.astatus == 'bad'), 'astatus'] = 'transl'
# print(ref_qua0.head())

Xq,Yq, _ = HTQ_get_xy(ref_qua0, class_col='astatus', features=None, scaling=1, select_mode='RFE')
# Xq45pca, totvar_q45, feats_q45 = HTQ_pca_transform(Xq, ref_qua0, 2, 'Dim1', print_best=15)
# nese_visualizePCA(Xq45pca, Yq, totvar_q45, dimx=1, dimy=2, feats=45)
crossvalidate(Xq,Yq, algo='SVM', grid=0, cv=10, class_weight = 'balanced')
print('Translationese is as clearly revealed in student translations, in fact, the SVM results are 2% better', file=sys.stderr)

# print('Reducing the feature set to the top 15 best translationese indicators does not degrade the results much', file=sys.stderr)

# Can PCA visualize distinctions between good and bad in the presence of non-translated Russian
# Xq3,Yq3, _ = HTQ_get_xy(ref_qua, class_col='astatus', features=None, scaling=1, select_mode='RFE')
# Xq3_45pca, totvar3_q45, feats3_q45 = HTQ_pca_transform(Xq3, ref_qua, 2, 'Dim1', print_best=0)
# HTQ_visualizePCA(Xq3_45pca, Yq3, totvar3_q45, dimx=1, dimy=2, feats=45)

# comparing en, ru_ref, bad, good
# ref_qua_en = pd.concat([ref,df], axis=0, join='outer', sort=False)
# print(ref_qua_en.shape)
# X4,Y4, df4 = HTQ_get_xy(ref_qua_en, class_col='astatus', features=None, scaling=1, select_mode='RFE')
# X4pca, totvar_4, feats_4 = HTQ_pca_transform(X4, df4, 2, 'Dim1', print_best=0)
# HTQ_visualizePCA(X4pca, Y4, totvar_4, dimx=1, dimy=2, feats=45)
# HTQ_textsdensity(X4pca[:,0], Y4, dimx=1, feats=45)

# # ***********
# # Step 1 of the research confirms that our feature set is relevant for translationese detection.
# #
# # We get up to 92-94% accuracy on the binary classification of translation and non-translations.
# #
# # The features that capture this dictinctions most rigorously (shared by both translational corpora) include: 'relativ', 'nnargs', 'sconj', 'whconj', 'but', 'ccomp', 'lexdens', 'correl', 'possdet', 'comp'
# # ***********

# if we want to quantify differences between values of a given variable in two given corpora:
quantify_diffs(ref_pro, feat='deverbals', corpus1='ref', corpus2='pro')

# Let's look what the 45 features are worth in terms of straightforward learning from good-bad labels
# If we are able to learn the good-bad distinctions: Are the same features get to the top best for best-worst as in translations-nontranslations??

# run the good/bad data thru the same pipeline
Xq45_2,Yq45_2, _ = HTQ_get_xy(df0, class_col='astatus', features=None, scaling=1, select_mode='RFE')
Xq45_2pca, totvar_q45_2, feats_q45_2 = HTQ_pca_transform(Xq45_2, df0, 2, 'Dim1', print_best=0)
# HTQ_visualizePCA(Xq45_2pca, Yq45_2, totvar_q45_2, dimx=1, dimy=2, feats=45)
crossvalidate(Xq45_2,Yq45_2, algo='SVM', grid=0, cv=10, class_weight = 'balanced')

crossvalidate(Xq45_2,Yq45_2, algo='dummy', grid=0, cv=10, class_weight = 'balanced')
HTQ_textsdensity(Xq45_2pca[:,0], Yq45_2, dimx=1, feats=5)
