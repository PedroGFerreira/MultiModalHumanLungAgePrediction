# MultiModalHumanLungAgePrediction

This repository provides the code and the supporting methods and results for the paper:
_"Predicting age from human lung tissue through multi-modal data integration"_
by Athos Moraes, Marta Moreno, Rogério Ribeiro & Pedro G. Ferreira


======
# Pipelines

## Gene Expression Pipeline

Pipeline for training and test of the gene expression model.

<img src="https://github.com/PedroGFerreira/MultiModalHumanLungAgePrediction/blob/main/geneexpression_pipeline.jpg"  width=70% height=70%>

## Methylation Pipeline

Pipeline for training and test of the DNA methylation model.

<img src="https://github.com/PedroGFerreira/MultiModalHumanLungAgePrediction/blob/main/methylation_pipeline.jpg"  width=70% height=70%>


Feature Selection and SMOGN on Methylation data. Application of the SMOGN data pre-processing following Branco et al [1] with the implementation from [2].

<img src="https://github.com/PedroGFerreira/MultiModalHumanLungAgePrediction/blob/main/methylation_featsel_smogn.jpg"  width=50%>

## Histological images Pipeline

UMAP based on the Haralick Features of 90 Lung samples. No clear separation is found on the age of the individuals based on the Haralick features.

<img src="https://github.com/PedroGFerreira/MultiModalHumanLungAgePrediction/blob/main/HR_UMAP.jpg"  width=50%>

Histological images Pipeline

<img src="https://github.com/PedroGFerreira/MultiModalHumanLungAgePrediction/blob/main/histological_pipeline.jpg"  width=70% height=70%>


CNN optimal Parameters. Parameters for the best performing model on the histological image regression.


<img src="https://github.com/PedroGFerreira/MultiModalHumanLungAgePrediction/blob/main/cnn_r2.jpg"  width=50%>


<img src="https://github.com/PedroGFerreira/MultiModalHumanLungAgePrediction/blob/main/CNN_parameters.jpg"  width=20%>




1. Branco, P., Torgo, L., Ribeiro, R.P.: Smogn: a pre-processing approach for im-
balanced regression. In: First international workshop on learning with imbalanced
domains: Theory and applications. pp. 36–50. PMLR (2017)

2. https://pypi.org/project/smogn/, Nick Kunz.
