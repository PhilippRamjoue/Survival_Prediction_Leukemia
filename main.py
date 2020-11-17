import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy


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

    columns_for_dummies = ['donor_ABO', 'recipient_age_int', 'recipient_ABO', 'disease', 'disease_group',
                           'HLA_match', 'HLA_group_1']


    columns_for_dropping = ['time_to_acute_GvHD_III_IV']

    # strong corrolation between missing CMV_status and recipient_CMV

    tmp_list = []
    for i in range(len(dataset)):
        tmp_counter = 0
        for col in dataset:
            tmp = dataset.at[i,col]
            if dataset.at[i,col] == '?':
                dataset.at[i, col] = np.NaN
                tmp_counter = tmp_counter + 1
        tmp_list.append(tmp_counter)


    # check out columns with the most nan values
    summed_nans_cols = (dataset.isna().sum()).sort_values(ascending=False)

    print(summed_nans_cols.head(10))

    print("%d %% of the 'time_to_acute_GvHD_III_IV ' data is nan" %((summed_nans_cols[0]/dataset.shape[0])*100))

    # Column 'time_to_acute_GvHD_III_IV' has to be dropped because 147 of 187 values are missing.
    dataset.drop(columns_for_dropping, inplace=True, axis=1)

    tmp_dataset = pd.get_dummies(dataset, columns=columns_for_dummies)

    summed_nans_cols = tmp_dataset.isna().sum()

    summed_nans_rows = tmp_dataset.isna().sum(axis=1)
    ax = plt.hist(summed_nans_rows)
  #  plt.show()
    print(summed_nans_cols.sort_values(ascending=False))
    #print(tmp_list.sort(reverse=True))
    #data['donor_age_below_35']=data.donor_age_below_35.apply(lambda s:1 if s=="yes" else 0)
    print("Test")

    #hier weiter
#'donor_CMV', #present/absent
#'recipient_age_below_10', #yes/no
#'recipient_gender', #male/female
#'recipient_rh',  #plus/minus
#'recipient_CMV', #present/absent
#'gender_match', #female_to_male/other
#'ABO_match', #machted/mismatched
#'HLA_mismatch', #matched/mismatched
#'risk_group', #high/low
#'stem_cell_source', #peripheral blood/bone marrow
#'tx_post_relapse', #yes/no
#'ANC_recovery', #yes/no
#'PLT_recovery', #yes/no
#'acute_GvHD_II_III_IV', #yes/no
#'acute_GvHD_III_IV', #yes/no
#'extensive_chronic_GvHD' #yes/no
#'relapse' #yes/no


#print(dataset['donor_age'])

#new = pd.get_dummies(dataset,columns=columns_for_dummies)
#print(new)

#max_value = max(tmp_scan_value_list)

#normalized_scan_value_list = [x / max_value for x in tmp_scan_value_list]
'''
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
encoded_customers = imputer.fit_transform(encoded_customers)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

customers_scaled = pd.DataFrame(scaler.fit_transform(encoded_customers.astype(float)))

print(customers_scaled.shape)
print(customers_scaled.head())

def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)

    return x_df, y_df
    
'''

def main():
    dataset = pd.read_csv('dataset/bone-marrow-dataset.csv', sep=',')

    dropna_data(dataset)

    clean_data(dataset)

if __name__ == '__main__':
    main()