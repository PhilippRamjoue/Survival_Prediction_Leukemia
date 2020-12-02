import requests
import json

scoring_uri = 'todo'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
              "donor_age_below_35": 1,
              "donor_CMV": 0,
              "recipient_age_below_10": 1,
              "recipient_gender": 1,
              "recipient_rh": 1,
              "recipient_CMV": 0,
              "gender_match": 1,
              "ABO_match": 0,
              "HLA_mismatch": 1,
              "risk_group": 1,
              "stem_cell_source": 1,
              "tx_post_relapse": 1,
              "ANC_recovery": 1,
              "PLT_recovery": 1,
              "acute_GvHD_II_III_IV": 0,
              "acute_GvHD_III_IV": 0,
              "relapse": 0,
              "donor_age": 0.2860763358,
              "recipient_age": 0.170212766,
              "recipient_body_mass": 0.0764082543,
              "CD34_x1e6_per_kg": 0.1030005264,
              "CD3_x1e8_per_kg": 0.0592638802,
              "CD3_to_CD34_ratio": 0.0642309858,
              "time_to_ANC_recovery": 0.4615384615,
              "time_to_PLT_recovery": 0.3198380567,
              "time_to_acute_GvHD_III_IV": 0.0,
              "donor_ABO_0": 0,
              "donor_ABO_A": 1,
              "donor_ABO_AB": 0,
              "donor_ABO_B": 0,
              "recipient_age_int_0_5": 1,
              "recipient_age_int_10_20": 0,
              "recipient_age_int_5_10": 0,
              "recipient_ABO_0": 1,
              "recipient_ABO_A": 0,
              "recipient_ABO_AB": 0,
              "recipient_ABO_B": 0,
              "disease_ALL": 0,
              "disease_AML": 0,
              "disease_chronic": 0,
              "disease_lymphoma": 0,
              "disease_nonmalignant": 1,
              "disease_group_malignant": 0,
              "disease_group_nonmalignant": 1,
              "CMV_status_0": 1, "CMV_status_1": 0,
              "CMV_status_2": 0, "CMV_status_3": 0,
              "HLA_match_10\/10": 1,
              "HLA_match_7\/10": 0,
              "HLA_match_8\/10": 0,
              "HLA_match_9\/10": 0,
              "antigen_0": 1,
              "antigen_1": 0,
              "antigen_2": 0,
              "antigen_3": 0,
              "allel_0": 1,
              "allel_1": 0,
              "allel_2": 0,
              "allel_3": 0,
              "allel_4": 0,
              "HLA_group_1_DRB1_cell": 0,
              "HLA_group_1_matched": 1,
              "HLA_group_1_mismatched": 0,
              "HLA_group_1_one_allel": 0,
              "HLA_group_1_one_antigen": 0,
              "HLA_group_1_three_diffs": 0,
              "HLA_group_1_two_diffs": 0
          }
      ]
    }



# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())