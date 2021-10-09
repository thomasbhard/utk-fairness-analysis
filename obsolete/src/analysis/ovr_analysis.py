"""
Perform a one vs rest analysis for predictions made by the gender-model
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

FILENAME = "df_predctions_all_ref.csv"
SAVE_PLOTS = False

pred_path = os.path.join(os.path.dirname(__file__), "..", "..", "predictions", FILENAME)
df = pd.read_csv(pred_path, index_col=0)

sns.set()
colors = ['#edf8e9','#c7e9c0','#a1d99b','#74c476','#31a354','#006d2c']



dataset_dict = {
    'race_id': {
        0: 'white', 
        1: 'black', 
        2: 'asian', 
        3: 'indian', 
        4: 'others'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    },
    'age_id': {
        0: '<10',
        1: '10-20',
        2: '20-30',
        3: '30-40',
        4: '40-60',
        5: '60-80',
        6: '80+'
    }
}

dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((r, i) for i, r in dataset_dict['race_id'].items())
dataset_dict['age_alias'] = dict((a, i) for i, a in dataset_dict['age_id'].items())






def pre_processing(df):
    """Remove unnecessary variables (age_pred, race_pred) and prepare df to work with the aif360 BLD
    """
    # analysis with regard to the gender prediction only -> dropping age and race predictions
    df_bld = df.drop(columns=['age_pred', 'race_pred']).rename(columns={'age_true': 'age', 'race_true': 'race'})

    # transforming gender_true and gender_pred into a single attribute if the prediction was correct
    pred_true = []
    for i, row in df_bld.iterrows():
        if(row['gender_true'] == row['gender_pred']):
            pred_true.append(1)
        else:
            pred_true.append(0)

    df_bld['pred_true'] = pred_true
    df_bld = df_bld.drop(columns=['gender_pred']).rename(columns={'gender_true': 'gender'})

    bins = [0, 10, 20, 30, 40, 60, 80, np.inf]
    # cutting the age into bins
    age_binned = pd.cut(df_bld['age'], bins, labels=[0,1,2,3,4,5,6])
    df_bld['age'] = age_binned

    return df_bld


def fairness_summary(bld_metric, plot=True, description=""):
    summary = {'base_rate_priviledged': bld_metric.base_rate(privileged=True),
     'base_rate_unpriviledged': bld_metric.base_rate(privileged=False),
     'consistency': bld_metric.consistency()[0],
     'disparate_impact': bld_metric.disparate_impact(),
     'mean_difference': bld_metric.mean_difference(),
     'smoothed_empirical_differential_fairness': bld_metric.smoothed_empirical_differential_fairness(concentration=1.0)}

    return summary


def one_vs_rest_grouping(df, conditions={'race': 1, 'gender': 1}):
    """ Add a group collum to the dataframe according to the given conditions
        The subgroup, the group for which all conditions apply has the label 1 the rest 0 
    """
    groups = []
    for i, row in df.iterrows():
        group = 1 # assume subgroup until on condition fails
        for column, value in conditions.items():
            if(row[column] != value):
                group = 0
                break
        groups.append(group)

    new_df = df.copy()
    new_df['group'] = groups

    return new_df


def plot_disparate_impact(df, title):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    groups = df['group']
    y_pos = np.arange(len(groups))
    disparate_impact = df['disparate_impact']

    ax.barh(y_pos, disparate_impact, align='center', color=colors[2])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Disparate impact')


    plt.xlim([0.6, 1.2])

    if SAVE_PLOTS:
        print(f"Saving: {title}")
        plt.savefig(f"Plots/disparate_impact/{title.lower().replace(' ', '_')}.png", dpi=300,bbox_inches='tight')
        plt.savefig(f"Plots/disparate_impact/{title.lower().replace(' ', '_')}.eps", format="eps",bbox_inches='tight')

    ax.set_title(title)
    plt.show()


def gender_analyis(df_bld):
    df_metrics = pd.DataFrame(columns=['group', 'base_rate_priviledged', 'base_rate_unpriviledged', 'consistency', 'disparate_impact', 'mean_difference', 'smoothed_empirical_differential_fairness'])
    for gender_id, gender_alias in dataset_dict['gender_id'].items():
        df_sub = one_vs_rest_grouping(df_bld, conditions={'gender': gender_id})
        bld = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=df_sub, label_names=['pred_true'], protected_attribute_names=['group'])
        bld_metric = BinaryLabelDatasetMetric(bld, unprivileged_groups=[{'group': 0}], privileged_groups=[{'group': 1}])
        summary = fairness_summary(bld_metric, description=gender_alias)
        summary['group'] = gender_alias
        df_metrics = df_metrics.append(summary, ignore_index=True)

    plot_disparate_impact(df_metrics, "Disparate impact gender")


def age_analysis(df_bld):
    df_metrics = pd.DataFrame(columns=['group', 'base_rate_priviledged', 'base_rate_unpriviledged', 'consistency', 'disparate_impact', 'mean_difference', 'smoothed_empirical_differential_fairness'])

    for age_id, age_alias in dataset_dict['age_id'].items():
        df_sub = one_vs_rest_grouping(df_bld, conditions={'age': age_id})
        bld = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=df_sub, label_names=['pred_true'], protected_attribute_names=['group'])
        bld_metric = BinaryLabelDatasetMetric(bld, unprivileged_groups=[{'group': 0}], privileged_groups=[{'group': 1}])
        summary = fairness_summary(bld_metric, description=age_alias)
        summary['group'] = age_alias
        df_metrics = df_metrics.append(summary, ignore_index=True)
    
    plot_disparate_impact(df_metrics, "Disparate impact age")


    












def ovr_analysis(df):
    df_bld = pre_processing(df)
    gender_analyis(df_bld)
    age_analysis(df_bld)



if __name__ == "__main__":
    ovr_analysis(df)
