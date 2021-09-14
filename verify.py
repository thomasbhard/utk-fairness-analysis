"""
Verify if the csv with the predictions matches the filereference file. The first predictions, prior to the random seed in the training should not match
"""
import pandas as pd

pred_df = pd.read_csv(r"Predictions\df_predctions_all_ref.csv")
file_df = pd.read_csv(r"Predictions\filereference.csv")

pred_df = pred_df.iloc[0:55]

assert len(pred_df) == len(
    file_df), f"Length not equal: {len(pred_df)} and {len(file_df)}"

for i, row in pred_df.iterrows():
    age_p = row['age_true']
    race_p = row['race_true']
    gender_p = row['gender_true']

    filename = file_df.loc[i]['filename']
    filename = filename.split('/')[1]
    age, gender, race, _ = filename.split('_')
    
    age, gender, race = int(age), int(gender), int(race)

    print((age, gender, race))

    assert age_p == age, f"Age does not match in line {i}"
    assert race_p == race, f"Race does not match in line {i}"
    assert gender_p == gender,  f"Gender does not match in line {i}"






