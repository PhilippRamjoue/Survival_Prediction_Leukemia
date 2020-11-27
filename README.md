# Survival prediction after bone marrow transplantation

*TODO:* Write a short introduction to your project.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

The chosen dataset is: __Bone marrow transplant: children Data set__ and can be found on [UCI](https://archive.ics.uci.edu/ml/datasets/Bone+marrow+transplant%3A+children#). Currently the dataset can't be downloaded directly form this site, I got it from the developers.

The source is owned by:

Marek Sikora (marek.sikora '@' polsl.pl), Lukasz Wróbel (lukasz.wrobel '@' polsl.pl), Adam Gudys´› (adam.gudys '@' polsl.pl)
Faculty of Automatic Control, Electronics and Computer Science, Silesian University of Technology, 44-100 Gliwice, Poland 
 
The relevant Paper is:

Sikora, M, Wróbel, L, Gudys´›, A (2019) GuideR: a guided separate-and-conquer rule learning in classification, regression, and survival settings,
Knowledge-Based Systems, 173:1-14 ([Web Link](https://www.sciencedirect.com/science/article/abs/pii/S0950705119300802?via%3Dihub))
 

### Overview

##### Data Set Information [[cited from here](https://archive.ics.uci.edu/ml/datasets/Bone+marrow+transplant%3A+children#)]

The data set describes pediatric patients with several hematologic diseases: malignant disorders
(i.a. acute lymphoblastic leukemia, acute myelogenous leukemia, chronic myelogenous leukemia, myelodysplastic syndrome) and nonmalignant cases (i.a. severe aplastic anemia, Fanconi anemia, with X-linked adrenoleukodystrophy). All patients were subject to the unmanipulated allogeneic unrelated donor hematopoietic stem cell transplantation.

The motivation of the study was to identify the most important factors influencing the success or failure of the
transplantation procedure. In particular, the aim was to verify the hypothesis that increased dosage of
CD34+ cells / kg extends overall survival time without simultaneous occurrence of undesirable events affecting patients' quality of life ([Kalwak et al., 2010](https://www.bbmt.org/article/S1083-8791(10)00148-5/fulltext)).

The data set has been used in our work concerning survival rules
([Wróbel et al., 2017](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1693-x))
and user-guided rule induction ([Sikora et al., 2019](https://www.sciencedirect.com/science/article/abs/pii/S0950705119300802?via%3Dihub)).
The authors of the research on stem cell transplantation ([Kalwak et al., 2010](https://www.bbmt.org/article/S1083-8791(10)00148-5/fulltext)) who inspired our study also contributed to the set.

### Task

In this project a model is developed that can predict the _survival_status_ based on the given dataset parameters. For this,
Azure Machine Learning is used to train a LogisticRegression model (optimized with Hyperdrive) and parallel get a optimized 
AutoML model. [AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml) offers the great
possibility to explore the model and get insights which parameters are most important and how they are distributed. It's
very interesting to check out if the above mentioned CD34+ cells/kg parameter has a big influence on the model.

### Access

The original provided dataset is an arff file from that I extracted the content and created a cvs dataset. Both files 
are in the dataset folder. To access the csv file in the code I use the "global" path.

```
dataset_path = 'https://raw.githubusercontent.com/PhilippRamjoue/Leukemia_Classification/main/dataset/bone-marrow-dataset.csv'

ds = Dataset.Tabular.from_delimited_files(path=dataset_path)
```

### Dataset Exploration

First thing to do is to explore the dataset. Following 39 categories are provided:

- __donor_age__ - Age of the donor at the time of hematopoietic stem cells apheresis
- __donor_age_below_35__ - Is donor age less than 35 (yes, no)
- __donor_ABO__ - ABO blood group of the donor of hematopoietic stem cells (0, A, B, AB)
- __donor_CMV__ - Presence of cytomegalovirus infection in the donor of hematopoietic stem cells prior to transplantation (present, absent)
- __recipient_age__ - Age of the recipient of hematopoietic stem cells at the time of transplantation
- __recipient_age_below_10__ - Is recipient age below 10 (yes, no)
- __recipient_age_int__ - Age of the recipient discretized to intervals (0,5], (5, 10], (10, 20]
- __recipient_gender__ - Gender of the recipient (female, male)
- __recipient_body_mass__ - Body mass of the recipient of hematopoietic stem cells at the time of the transplantation
- __recipient_ABO__ - ABO blood group of the recipient of hematopoietic stem cells (0, A, B, AB)
- __recipient_rh__ - Presence of the Rh factor on recipient’s red blood cells (plus, minus)
- __recipient_CMV__ - Presence of cytomegalovirus infection in the donor of hematopoietic stem cells prior to transplantation (present, absent)
- __disease__ - Type of disease (ALL, AML, chronic, nonmalignant, lymphoma)
- __disease_group__ - Type of disease (malignant, nonmalignant)
- __gender_match__ - Compatibility of the donor and recipient according to their gender (female to male, other)
- __ABO_match__ - Compatibility of the donor and the recipient of hematopoietic stem cells according to ABO blood group (matched, mismatched)
- __CMV_status__ - Serological compatibility of the donor and the recipient of hematopoietic stem cells according to cytomegalovirus infection prior to transplantation (the higher the value, the lower the compatibility)
- __HLA_match__ - Compatibility of antigens of the main histocompatibility complex of the donor and the recipient of hematopoietic stem cells (10/10, 9/10, 8/10, 7/10)
- __HLA_mismatch__ - HLA matched or mismatched
- __antigen__ - In how many antigens there is a difference between the donor and the recipient (0-3)
- __allel__ - In how many allele there is a difference between the donor and the recipient (0-4)
- __HLA_group_1__ - The difference type between the donor and the recipient (HLA matched, one antigen, one allel, DRB1 cell, two allele or allel+antigen, two antigenes+allel, mismatched)
- __risk_group__ - Risk group (high, low)
- __stem_cell_source__ - Source of hematopoietic stem cells (peripheral blood, bone marrow)
- __tx_post_relapse__ - The second bone marrow transplantation after relapse (yes ,no)
- __CD34_x1e6_per_kg__ - CD34kgx10d6 - CD34+ cell dose per kg of recipient body weight (10^6/kg)
- __CD3_x1e8_per_kg__ - CD3+ cell dose per kg of recipient body weight (10^8/kg)
- __CD3_to_CD34_ratio__ - CD3+ cell to CD34+ cell ratio
- __ANC_recovery__ - Neutrophils recovery defined as neutrophils count >0.5 x 10^9/L (yes, no)
- __time_to_ANC_recovery__ - Time in days to neutrophils recovery
- __PLT_recovery__ - Platelet recovery defined as platelet count >50000/mm3 (yes, no)
- __time_to_PLT_recovery__ - Time in days to platelet recovery
- __acute_GvHD_II_III_IV__ - Development of acute graft versus host disease stage II or III or IV (yes, no)
- __acute_GvHD_III_IV__ - Development of acute graft versus host disease stage III or IV (yes, no)
- __time_to_acute_GvHD_III_IV__ - Time in days to development of acute graft versus host disease stage III or IV
- __extensive_chronic_GvHD__ - Development of extensive chronic graft versus host disease (yes, no)
- __relapse__ - Relapse of the disease (yes, no)
- __survival_time__ - Time of observation (if alive) or time to event (if dead) in days
- __survival_status__ - Survival status (0 - alive, 1 - dead)

In sum there are 189 entries with each 39 columns.



### Cleaning

#### Version 1

#### Version 2


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
