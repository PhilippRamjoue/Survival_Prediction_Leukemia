import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler


#  donor_age - Age of the donor at the time of hematopoietic stem cells apheresis
#  donor_age_below_35 - Is donor age less than 35 (yes, no)
#  donor_ABO - ABO blood group of the donor of hematopoietic stem cells (0, A, B, AB)
#  donor_CMV - Presence of cytomegalovirus infection in the donor of hematopoietic stem cells prior to transplantation (present, absent)
#  recipient_age - Age of the recipient of hematopoietic stem cells at the time of transplantation
#  recipient_age_below_10 - Is recipient age below 10 (yes, no)
#  recipient_age_int - Age of the recipient discretized to intervals (0,5], (5, 10], (10, 20]
#  recipient_gender - Gender of the recipient (female, male)
#  recipient_body_mass - Body mass of the recipient of hematopoietic stem cells at the time of the transplantation
#  recipient_ABO - ABO blood group of the recipient of hematopoietic stem cells (0, A, B, AB)
#  recipient_rh - Presence of the Rh factor on recipient’s red blood cells (plus, minus)
#  recipient_CMV - Presence of cytomegalovirus infection in the donor of hematopoietic stem cells prior to transplantation (present, absent)
#  disease - Type of disease (ALL, AML, chronic, nonmalignant, lymphoma)
#  disease_group - Type of disease (malignant, nonmalignant)
#  gender_match - Compatibility of the donor and recipient according to their gender (female to male, other)
#  ABO_match - Compatibility of the donor and the recipient of hematopoietic stem cells according to ABO blood group (matched, mismatched)
#  CMV_status - Serological compatibility of the donor and the recipient of hematopoietic stem cells according to cytomegalovirus infection prior to transplantation (the higher the value, the lower the compatibility)
#  HLA_match - Compatibility of antigens of the main histocompatibility complex of the donor and the recipient of hematopoietic stem cells (10/10, 9/10, 8/10, 7/10)
#  HLA_mismatch - HLA matched or mismatched
#  antigen - In how many antigens there is a difference between the donor and the recipient (0-3)
#  allel - In how many allele there is a difference between the donor and the recipient (0-4)
#  HLA_group_1 - The difference type between the donor and the recipient (HLA matched, one antigen, one allel, DRB1 cell, two allele or allel+antigen, two antigenes+allel, mismatched)
#  risk_group - Risk group (high, low)
#  stem_cell_source - Source of hematopoietic stem cells (peripheral blood, bone marrow)
#  tx_post_relapse - The second bone marrow transplantation after relapse (yes ,no)
#  CD34_x1e6_per_kg - CD34kgx10d6 - CD34+ cell dose per kg of recipient body weight (10^6/kg)
#  CD3_x1e8_per_kg - CD3+ cell dose per kg of recipient body weight (10^8/kg)
#  CD3_to_CD34_ratio - CD3+ cell to CD34+ cell ratio
#  ANC_recovery - Neutrophils recovery defined as neutrophils count >0.5 x 10^9/L (yes, no)
#  time_to_ANC_recovery - Time in days to neutrophils recovery
#  PLT_recovery - Platelet recovery defined as platelet count >50000/mm3 (yes, no)
#  time_to_PLT_recovery - Time in days to platelet recovery
#  acute_GvHD_II_III_IV - Development of acute graft versus host disease stage II or III or IV (yes, no)
#  acute_GvHD_III_IV - Development of acute graft versus host disease stage III or IV (yes, no)
#  time_to_acute_GvHD_III_IV - Time in days to development of acute graft versus host disease stage III or IV
#  extensive_chronic_GvHD - Development of extensive chronic graft versus host disease (yes, no)
#  relapse - Relapse of the disease (yes, no)
#  survival_time - Time of observation (if alive) or time to event (if dead) in days
#  survival_status - Survival status (0 - alive, 1 - dead)

def dropna_data(org_dataset):

    tmp_data = copy.deepcopy(org_dataset)

    shape_org = tmp_data.shape

    for i in range(len(tmp_data)):
        for col in tmp_data:
            if tmp_data.at[i,col] == '?':
                tmp_data.at[i, col] = np.NaN

    dropped_columns_frame = tmp_data.dropna(axis=1)

    dropped_rows_frame = tmp_data.dropna(axis=0)

    shape_col_drop = dropped_columns_frame.shape
    col_drop_loss = 100 - ((shape_col_drop[1]/shape_org[1]) * 100)

    shape_row_drop = dropped_rows_frame.shape
    row_drop_loss = 100 - ((shape_row_drop[0] / shape_org[0]) * 100)

    print("Shape of original frame: %s" % str(shape_org))
    print("Shape of frame with dropped nan columns: %s; Loss: %d %%" % (str(shape_col_drop), col_drop_loss))
    print("Shape of frame with dropped nan rows: %s; Loss: %d %%" % (str(shape_row_drop), row_drop_loss))


def clean_data(org_dataset):

    dataset = copy.deepcopy(org_dataset)

    columns_for_dropping = ['extensive_chronic_GvHD']

    # 1. Column 'extensive_chronic_GvHD' can be dropped because there are 31 of 187 values are missing
    dataset.drop(columns_for_dropping, inplace=True, axis=1)

    # 2. replace all ? in this column with 0.0
    # ? are used if the related conditions (PLT_recovery, ANC_recovery and acute_GvHD_III_IV) are 'no' so there is
    # no time passed until the event. In this cases the value 0.0 is also suitable instead of the ?
    dataset['time_to_PLT_recovery'] = dataset.time_to_PLT_recovery.apply(lambda s: '0.0' if s == "?" else s)

    dataset['time_to_ANC_recovery'] = dataset.time_to_ANC_recovery.apply(lambda s: '0.0' if s == "?" else s)

    dataset['time_to_acute_GvHD_III_IV'] = dataset.time_to_acute_GvHD_III_IV.apply(lambda s: '0.0' if s == "?" else s)

    tmp_list = []
    for i in range(len(dataset)):
        tmp_counter = 0
        for col in dataset:
            if dataset.at[i,col] == '?':
                dataset.at[i, col] = np.NaN
                tmp_counter = tmp_counter + 1
        tmp_list.append(tmp_counter)

    shape_org = dataset.shape

    dataset.dropna(axis=0,inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    shape_row_drop = dataset.shape
    row_drop_loss = 100 - ((shape_row_drop[0] / shape_org[0]) * 100)

    print("Shape of original frame: %s" % str(shape_org))
    print("Shape of frame with dropped nan rows: %s; Loss: %d %%" % (str(shape_row_drop), row_drop_loss))

    # Convert categorical categories to numbers

    # 'donor_age_below_35', #yes/no
    dataset['donor_age_below_35'] = dataset.donor_age_below_35.apply(lambda s: 1 if s == "yes" else 0)

    # 'donor_CMV', #present/absent
    dataset['donor_CMV'] = dataset.donor_CMV.apply(
        lambda s: 1 if s == "present" else 0)

    # 'recipient_age_below_10', #yes/no
    dataset['recipient_age_below_10'] = dataset.recipient_age_below_10.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'recipient_gender', #male/female
    dataset['recipient_gender'] = dataset.recipient_gender.apply(
        lambda s: 1 if s == "male" else 0)

    # 'recipient_rh',  #plus/minus
    dataset['recipient_rh'] = dataset.recipient_rh.apply(
        lambda s: 1 if s == "plus" else 0)

    # 'recipient_CMV', #present/absent
    dataset['recipient_CMV'] = dataset.recipient_CMV.apply(
        lambda s: 1 if s == "present" else 0)

    # 'gender_match', #female_to_male/other
    dataset['gender_match'] = dataset.gender_match.apply(
        lambda s: 1 if s == "female_to_male" else 0)

    # 'ABO_match', #matched/mismatched
    dataset['ABO_match'] = dataset.ABO_match.apply(
        lambda s: 1 if s == "matched" else 0)

    # 'HLA_mismatch', #matched/mismatched
    dataset['HLA_mismatch'] = dataset.HLA_mismatch.apply(
        lambda s: 1 if s == "matched" else 0)

    # 'risk_group', #high/low
    dataset['risk_group'] = dataset.risk_group.apply(
        lambda s: 1 if s == "high" else 0)

    # 'stem_cell_source', #peripheral_blood/bone_marrow
    dataset['stem_cell_source'] = dataset.stem_cell_source.apply(
        lambda s: 1 if s == "bone_marrow" else 0)

    # 'tx_post_relapse', #yes/no
    dataset['tx_post_relapse'] = dataset.tx_post_relapse.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'ANC_recovery', #yes/no
    dataset['ANC_recovery'] = dataset.ANC_recovery.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'PLT_recovery', #yes/no
    dataset['PLT_recovery'] = dataset.PLT_recovery.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'acute_GvHD_II_III_IV', #yes/no
    dataset['acute_GvHD_II_III_IV'] = dataset.acute_GvHD_II_III_IV.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'acute_GvHD_III_IV', #yes/no
    dataset['acute_GvHD_III_IV'] = dataset.acute_GvHD_III_IV.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'relapse' #yes/no
    dataset['relapse'] = dataset.relapse.apply(
        lambda s: 1 if s == "yes" else 0)

    list_for_minmax_scaling = ['donor_age','recipient_age','recipient_body_mass','CD34_x1e6_per_kg','CD3_x1e8_per_kg',
                               'CD3_to_CD34_ratio','time_to_ANC_recovery','time_to_PLT_recovery',
                               'time_to_acute_GvHD_III_IV','survival_time']

    scaler = MinMaxScaler()

    dataset_scaled =  pd.DataFrame(scaler.fit_transform(dataset[list_for_minmax_scaling]),columns=list_for_minmax_scaling)

    dataset.drop(columns=list_for_minmax_scaling, inplace=True, axis=1)

    dataset = dataset.join(dataset_scaled)

    # Dummy encoding
    list_for_dummy_encoding = ['donor_ABO','recipient_age_int','recipient_ABO','disease','disease_group',
                               'CMV_status','HLA_match','antigen','allel','HLA_group_1']

    dataset_finalized = pd.get_dummies(dataset, columns=list_for_dummy_encoding)

    label = dataset_finalized.pop("survival_status")
    #dataset_finalized = dataset_finalized.join(label)

    return dataset_finalized, label


def clean_data_v2(org_dataset):

    dataset = copy.deepcopy(org_dataset)

    # With reference to the 5-years-survival rate in medical context there are 3 categories:
    # survival_status_0: still alive and over 5 years in observation
    # survival_status_1: unfortunately dead
    # survival_status_ongoing: still alive and under 5 years in observation

    survival_status_0 = 0
    survival_status_1 = 0
    survival_status_ongoing = 0
    five_years_in_days = 365 * 5
    index_for_droping = []
    for i in range(len(dataset)):

        tmp_time = dataset.loc[i,'survival_time']
        tmp_status = dataset.loc[i, 'survival_status']

        if(tmp_time > five_years_in_days) and (0 == tmp_status):
            survival_status_0 = survival_status_0 + 1
        elif (tmp_time < five_years_in_days) and (1 == tmp_status):
            survival_status_1 = survival_status_1 + 1
        elif (tmp_time < five_years_in_days) and (0 == tmp_status):
            survival_status_ongoing = survival_status_ongoing +1
            index_for_droping.append(i)
        else:
            # there sould not be another type
            pass

    print("survival_status_0: %d" %survival_status_0)
    print("survival_status_1: %d" % survival_status_1)
    print("survival_status_ongoing: %d" % survival_status_ongoing)

    # 0. Drop rows with "ongoing" survival_status
    dataset.drop(index=index_for_droping, inplace=True, axis=0)
    dataset.reset_index(drop=True, inplace=True)

    columns_for_dropping = ['extensive_chronic_GvHD','survival_time']

    # 1. Column 'extensive_chronic_GvHD' can be dropped because there are 31 of 121 values are missing.
    # Column 'survival_time' can be dropped because the label 'survival_status' implicitly includes this feature.
    dataset.drop(columns_for_dropping, inplace=True, axis=1)

    # 2. replace all ? in this column with 0.0
    # ? are used if the related conditions (PLT_recovery, ANC_recovery and acute_GvHD_III_IV) are 'no' so there is
    # no time passed until the event. In this cases the value 0.0 is also suitable instead of the ?
    dataset['time_to_PLT_recovery'] = dataset.time_to_PLT_recovery.apply(lambda s: '0.0' if s == "?" else s)

    dataset['time_to_ANC_recovery'] = dataset.time_to_ANC_recovery.apply(lambda s: '0.0' if s == "?" else s)

    dataset['time_to_acute_GvHD_III_IV'] = dataset.time_to_acute_GvHD_III_IV.apply(lambda s: '0.0' if s == "?" else s)

    tmp_list = []
    for i in range(len(dataset)):
        tmp_counter = 0
        for col in dataset:
            if dataset.at[i,col] == '?':
                dataset.at[i, col] = np.NaN
                tmp_counter = tmp_counter + 1
        tmp_list.append(tmp_counter)

    shape_org = dataset.shape

    dataset.dropna(axis=0,inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    shape_row_drop = dataset.shape
    row_drop_loss = 100 - ((shape_row_drop[0] / shape_org[0]) * 100)

    print("Shape of original frame: %s" % str(shape_org))
    print("Shape of frame with dropped nan rows: %s; Loss: %d %%" % (str(shape_row_drop), row_drop_loss))

    # Convert categorical categories to numbers

    # 'donor_age_below_35', #yes/no
    dataset['donor_age_below_35'] = dataset.donor_age_below_35.apply(lambda s: 1 if s == "yes" else 0)

    # 'donor_CMV', #present/absent
    dataset['donor_CMV'] = dataset.donor_CMV.apply(
        lambda s: 1 if s == "present" else 0)

    # 'recipient_age_below_10', #yes/no
    dataset['recipient_age_below_10'] = dataset.recipient_age_below_10.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'recipient_gender', #male/female
    dataset['recipient_gender'] = dataset.recipient_gender.apply(
        lambda s: 1 if s == "male" else 0)

    # 'recipient_rh',  #plus/minus
    dataset['recipient_rh'] = dataset.recipient_rh.apply(
        lambda s: 1 if s == "plus" else 0)

    # 'recipient_CMV', #present/absent
    dataset['recipient_CMV'] = dataset.recipient_CMV.apply(
        lambda s: 1 if s == "present" else 0)

    # 'gender_match', #female_to_male/other
    dataset['gender_match'] = dataset.gender_match.apply(
        lambda s: 1 if s == "female_to_male" else 0)

    # 'ABO_match', #matched/mismatched
    dataset['ABO_match'] = dataset.ABO_match.apply(
        lambda s: 1 if s == "matched" else 0)

    # 'HLA_mismatch', #matched/mismatched
    dataset['HLA_mismatch'] = dataset.HLA_mismatch.apply(
        lambda s: 1 if s == "matched" else 0)

    # 'risk_group', #high/low
    dataset['risk_group'] = dataset.risk_group.apply(
        lambda s: 1 if s == "high" else 0)

    # 'stem_cell_source', #peripheral_blood/bone_marrow
    dataset['stem_cell_source'] = dataset.stem_cell_source.apply(
        lambda s: 1 if s == "bone_marrow" else 0)

    # 'tx_post_relapse', #yes/no
    dataset['tx_post_relapse'] = dataset.tx_post_relapse.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'ANC_recovery', #yes/no
    dataset['ANC_recovery'] = dataset.ANC_recovery.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'PLT_recovery', #yes/no
    dataset['PLT_recovery'] = dataset.PLT_recovery.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'acute_GvHD_II_III_IV', #yes/no
    dataset['acute_GvHD_II_III_IV'] = dataset.acute_GvHD_II_III_IV.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'acute_GvHD_III_IV', #yes/no
    dataset['acute_GvHD_III_IV'] = dataset.acute_GvHD_III_IV.apply(
        lambda s: 1 if s == "yes" else 0)

    # 'relapse' #yes/no
    dataset['relapse'] = dataset.relapse.apply(
        lambda s: 1 if s == "yes" else 0)

    list_for_minmax_scaling = ['donor_age','recipient_age','recipient_body_mass','CD34_x1e6_per_kg','CD3_x1e8_per_kg',
                               'CD3_to_CD34_ratio','time_to_ANC_recovery','time_to_PLT_recovery',
                               'time_to_acute_GvHD_III_IV']

    scaler = MinMaxScaler()

    dataset_scaled =  pd.DataFrame(scaler.fit_transform(dataset[list_for_minmax_scaling]),columns=list_for_minmax_scaling)

    dataset.drop(columns=list_for_minmax_scaling, inplace=True, axis=1)

    dataset = dataset.join(dataset_scaled)

    # Dummy encoding
    list_for_dummy_encoding = ['donor_ABO','recipient_age_int','recipient_ABO','disease','disease_group',
                               'CMV_status','HLA_match','antigen','allel','HLA_group_1']

    dataset_finalized = pd.get_dummies(dataset, columns=list_for_dummy_encoding)

    label = dataset_finalized.pop("survival_status")
    #dataset_finalized = dataset_finalized.join(label)

    return dataset_finalized, label



def dataset_exploration(org_dataset):

    dataset = copy.deepcopy(org_dataset)

    # 1. Convert ? to NaN values
    for i in range(len(dataset)):
        for col in dataset:
            if dataset.at[i,col] == '?':
                dataset.at[i, col] = np.NaN

    # check out columns with the most nan values
    summed_nans_cols = (dataset.isna().sum()).sort_values(ascending=False)

    print(summed_nans_cols.head(20))

    print(
        "%d %% of the 'time_to_acute_GvHD_III_IV ' data is nan" % ((summed_nans_cols[0] / dataset.shape[0]) * 100))

def main():
    dataset = pd.read_csv('dataset/bone-marrow-dataset.csv', sep=',')

    dataset_exploration(dataset)

    dropna_data(dataset)

    clean_data_v2(dataset)

if __name__ == '__main__':
    main()