# HiT-IT
code and data related to translationese-for-quality project, presented at HiT-IT (RANLP 2019)

## Goal
This is a spin-off of the major project aimed at revealing translationese tendencies in out-of-English translations in German and Russian (2 targets project stared in Saarbrucken in Oct 2018).
The major goal of this study is to test the claim that translationese features can also be used as human translation quality indicators.

## Data
In this attempt we relied on the Russian translations labeled for quality in real-life situations of translation competitions and exams by translation experts (professional translators or translation teachers). 
We use the translations to the same source text that are in sharp contrast in terms of quality: they are either highest-ranking and prize-winning texts or those that received the lowest rankngs or 'fail' grade. 
To test our set of features for translationese detection we use a collection of genre-comparable professional translations as well as all student translations at our disposal (regardless of quality class) and a genre-comparable set of originally-authored Russian texts from the Russian National Corpus. 
All texts are newspaper texts of informative and argumentative nature. 

## Features
Our features describe statistical morpho-syntactic properties of texts and partly intersect with the register features, but also include known or suspected translationese indicators such as correlative construcions and degrees of nominalisation. 

## Method
SVM-based binary classification and RFE-based feature selection; PCA for visualisation

## Results
The 45 features that we have engenieered and extracted give us the **macro-averaged F1 score of over 0.92** on the translations vs. non-translations classification for both translational collections.
However, **the quality classes are barely indistinquishable** based on the same feature set (F1 = 0.64). 
It means that if the quality assessment labels bear any relation to the linguistic features of the texts, these features lie outside morpho-syntactic translationese properties of the texts. 
They might be of lexical or pragmatic/textual nature or related to accuracy rather than fluency of the texts. 

P.S. we are not even sure that translationese is immediately related to fluency (as in readability), for that matter. 
P.S. If need be, refer to 2target repository for preprocessing and tagging scripts and details.
