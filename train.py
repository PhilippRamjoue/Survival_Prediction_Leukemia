from sklearn.linear_model import LogisticRegression
import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler

run = Run.get_context()


def clean_data(org_dataset):

    # 0. Convert TabularData to pandas Dataframe
    dataset = org_dataset.to_pandas_dataframe()

    # With reference to the 5-years-survival rate in medical context there are 3 categories:
    # survival_status_0: still alive and over 5 years in observation
    # survival_status_1: unfortunately dead
    # survival_status_ongoing: still alive and under 5 years in observation

    survival_status_0 = 0
    survival_status_1 = 0
    survival_status_ongoing = 0
    five_years_in_days = 365 * 5
    index_for_dropping = []
    for i in range(len(dataset)):

        tmp_time = dataset.loc[i, 'survival_time']
        tmp_status = dataset.loc[i, 'survival_status']

        if (tmp_time > five_years_in_days) and (0 == tmp_status):
            survival_status_0 = survival_status_0 + 1
        elif (tmp_time < five_years_in_days) and (1 == tmp_status):
            survival_status_1 = survival_status_1 + 1
        elif (tmp_time < five_years_in_days) and (0 == tmp_status):
            survival_status_ongoing = survival_status_ongoing + 1
            index_for_dropping.append(i)
        else:
            # there sould not be another type
            pass

    print("survival_status_0: %d" % survival_status_0)
    print("survival_status_1: %d" % survival_status_1)
    print("survival_status_ongoing: %d" % survival_status_ongoing)

    # 0. Drop rows with "ongoing" survival_status
    dataset.drop(index=index_for_dropping, inplace=True, axis=0)
    dataset.reset_index(drop=True, inplace=True)

    columns_for_dropping = ['extensive_chronic_GvHD', 'survival_time']

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
            if dataset.at[i, col] == '?':
                dataset.at[i, col] = np.NaN
                tmp_counter = tmp_counter + 1
        tmp_list.append(tmp_counter)

    shape_org = dataset.shape

    dataset.dropna(axis=0, inplace=True)
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

    list_for_minmax_scaling = ['donor_age', 'recipient_age', 'recipient_body_mass', 'CD34_x1e6_per_kg',
                               'CD3_x1e8_per_kg',
                               'CD3_to_CD34_ratio', 'time_to_ANC_recovery', 'time_to_PLT_recovery',
                               'time_to_acute_GvHD_III_IV']

    scaler = MinMaxScaler()

    dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset[list_for_minmax_scaling]),
                                  columns=list_for_minmax_scaling)

    dataset.drop(columns=list_for_minmax_scaling, inplace=True, axis=1)

    dataset = dataset.join(dataset_scaled)

    # Dummy encoding
    list_for_dummy_encoding = ['donor_ABO', 'recipient_age_int', 'recipient_ABO', 'disease', 'disease_group',
                               'CMV_status', 'HLA_match', 'antigen', 'allel', 'HLA_group_1']

    dataset_finalized = pd.get_dummies(dataset, columns=list_for_dummy_encoding)

    label = dataset_finalized.pop("survival_status")
    # dataset_finalized = dataset_finalized.join(label)

    return dataset_finalized, label

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    # Data is located at:
    dataset_path = 'https://raw.githubusercontent.com/PhilippRamjoue/Leukemia_Classification/main/dataset/bone-marrow-dataset.csv'

    ds = Dataset.Tabular.from_delimited_files(path=dataset_path)

    x, y = clean_data(ds)

    # Split data into train and test sets.
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=42)

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

    run.log("accuracy", np.float(accuracy))

    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model,'outputs/model.joblib')


if __name__ == '__main__':
    main()