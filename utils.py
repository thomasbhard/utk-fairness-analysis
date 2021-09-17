from IPython.display import set_matplotlib_formats
import os
import glob
import math

from IPython.display import display, Markdown, Latex
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# PLOTTING SETTUP

set_matplotlib_formats('svg')
colors = ['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#31a354', '#006d2c']

IM_WIDTH = IM_HEIGHT = 198

# DATASET DICT

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

dataset_dict['gender_alias'] = dict(
    (g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((r, i)
                                  for i, r in dataset_dict['race_id'].items())
dataset_dict['age_alias'] = dict((a, i)
                                 for i, a in dataset_dict['age_id'].items())


def parse_dataset(dataset_path, ext='jpg'):
    """
    Used to extract information about our dataset. It does iterate over all images and return a DataFrame with
    the data (age, gender and sex) of all files.
    """
    def parse_info_from_file(path):
        """
        Parse information from a single file
        """
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')

            return int(age), dataset_dict['gender_id'][int(gender)], dataset_dict['race_id'][int(race)]
        except Exception as ex:
            return None, None, None

    files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))

    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)

    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'race', 'file']
    df = df.dropna()

    return df


def preprocess_image(img_path):
    """
    Used to perform some minor preprocessing on the image before inputting into the network.
    """
    im = Image.open(img_path)
    im = im.resize((IM_WIDTH, IM_HEIGHT))
    im = np.array(im) / 255.0

    return im


def plot_select_images(df, n=16, include_prediction=False):
    df_selection = df.sample(n, random_state=1).reset_index()
    n_cols = 4
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 17))
    for i, row in df_selection.iterrows():
        ax = axes.flat[i]
        ax.imshow(preprocess_image(row['file']))

        cur_gender = row['gender']
        #cur_gender_true = gender_true[img_idx]

        cur_age = row['age']
        cur_race = row['race']

        # age_threshold = 10
        # if cur_gender_pred == cur_gender_true and cur_race_pred == cur_race_true and abs(cur_age_pred - cur_age_true) <= age_threshold:
        #     ax.xaxis.label.set_color('green')
        # elif cur_gender_pred != cur_gender_true and cur_race_pred != cur_race_true and abs(cur_age_pred - cur_age_true) > age_threshold:
        #     ax.xaxis.label.set_color('red')

        # ax.set_xlabel('a: {}, g: {}, r: {}'.format(int(age_pred[img_idx]),
        #                         dataset_dict['gender_id'][gender_pred[img_idx]],
        #                         dataset_dict['race_id'][race_pred[img_idx]]))

        ax.set_title('a: {}, g: {}, r: {}'.format(round(cur_age),
                                                  cur_gender,
                                                  cur_race))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()


def plot_gender_distribution(df, title="Gender distribution"):
    gender = df['gender']

    labels = gender.value_counts().sort_index().index.tolist()
    counts = gender.value_counts().sort_index().values.tolist()

    # print(labels)
    # print(counts)

    pie, ax = plt.subplots(figsize=[10, 6])
    plt.pie(counts, labels=labels, autopct="%.1f%%", explode=[
            0.01]*2, pctdistance=0.5, colors=[colors[1], colors[-2]], startangle=90)

    # if SAVE_PLOTS:
    #     print(f"Saving: {title}")
    #     plt.savefig(f"Plots/{title.lower().replace(' ', '_')}.png", dpi=300,bbox_inches='tight')
    #     plt.savefig(f"Plots/{title.lower().replace(' ', '_')}.eps", format="eps",bbox_inches='tight')

    plt.title(title, fontsize=14)
    # plt.show()


def plot_race_distribution(df, title="Race distribution"):
    race = df['race']

    labels = race.value_counts().sort_index().index.tolist()
    counts = race.value_counts().sort_index().values.tolist()

    # print(labels)
    # print(counts)

    pie, ax = plt.subplots(figsize=[10, 6])
    plt.pie(counts, labels=labels, autopct="%.1f%%", explode=[
            0.01]*5, pctdistance=0.5, colors=colors[1:], startangle=90)

    # if SAVE_PLOTS:
    #     print(f"Saving: {title}")
    #     plt.savefig(f"Plots/{title.lower().replace(' ', '_')}.png", dpi=300,bbox_inches='tight')
    #     plt.savefig(f"Plots/{title.lower().replace(' ', '_')}.eps", format="eps",bbox_inches='tight')

    plt.title(title, fontsize=14)
    # plt.show()


def plot_age_distribution(df, title="Age distribution", bins=20):
    sns.set(rc={'axes.facecolor': colors[0], 'figure.facecolor': 'white'})

    plt.hist(df['age'], bins=bins, color=colors[-1])

    # if SAVE_PLOTS:
    #     print(f"Saving: {title}")
    #     plt.savefig(f"Plots/{title.lower().replace(' ', '_')}.png", dpi=300,bbox_inches='tight')
    #     plt.savefig(f"Plots/{title.lower().replace(' ', '_')}.eps", format="eps",bbox_inches='tight')

    plt.title(title, fontsize=14)
    # plt.show()


def plot_distributions(df):

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    axes = [ax1, ax2, ax3]

    # Gender

    gender = df['gender']
    labels = gender.value_counts().sort_index().index.tolist()
    counts = gender.value_counts().sort_index().values.tolist()
    ax1.pie(counts, labels=labels, autopct="%.1f%%", explode=[
            0.01]*2, pctdistance=0.5, colors=[colors[1], colors[-2]], startangle=90)
    ax1.set_title('Gender distribution')

    # Race

    race = df['race']
    labels = race.value_counts().sort_index().index.tolist()
    counts = race.value_counts().sort_index().values.tolist()
    ax2.pie(counts, labels=labels, autopct="%.1f%%", explode=[
            0.01]*5, pctdistance=0.6, colors=colors[1:], startangle=90)
    ax2.set_title('Race distribution')

    # Age

    sns.set(rc={'axes.facecolor': colors[0], 'figure.facecolor': 'white'})
    ax3.hist(df['age'], bins=20, color=colors[-1])
    ax3.set_title('Age distribution')
    ax3.set_xlabel("Age")
    ax3.set_ylabel("Number of images")

    plt.tight_layout(w_pad=2, h_pad=2)


def plot_gender_distribution_by_race(df):

    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax4 = plt.subplot2grid((2, 3), (1, 0))
    ax5 = plt.subplot2grid((2, 3), (1, 1))

    axes = [ax1, ax2, ax3, ax4, ax5]

    for ax, race in zip(axes, dataset_dict['race_alias'].keys()):
        df_race = df.loc[df['race'] == race]

        gender = df_race['gender']
        labels = gender.value_counts().sort_index().index.tolist()
        counts = gender.value_counts().sort_index().values.tolist()
        ax.pie(counts, labels=labels, autopct="%.1f%%", explode=[
               0.01]*2, pctdistance=0.5, colors=[colors[1], colors[-2]], startangle=90)
        ax.set_title(f"{race}")

    plt.tight_layout()


def plot_gender_distribution_by_age(df):

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 3), (0, 0))
    ax2 = plt.subplot2grid((3, 3), (0, 1))
    ax3 = plt.subplot2grid((3, 3), (0, 2))
    ax4 = plt.subplot2grid((3, 3), (1, 0))
    ax5 = plt.subplot2grid((3, 3), (1, 1))
    ax6 = plt.subplot2grid((3, 3), (1, 2))
    ax7 = plt.subplot2grid((3, 3), (2, 0))

    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

    bins = [0, 10, 20, 30, 40, 60, 80, np.inf]
    # cutting the age into bins
    age_binned = pd.cut(df['age'], bins, labels=[0, 1, 2, 3, 4, 5, 6])
    df_binned = df.copy()
    df_binned['age_binned'] = age_binned

    for ax, age in zip(axes, dataset_dict['age_alias'].keys()):
        df_age = df_binned.loc[df_binned['age_binned'] == dataset_dict['age_alias'][age]]

        gender = df_age['gender']
        labels = gender.value_counts().sort_index().index.tolist()
        counts = gender.value_counts().sort_index().values.tolist()
        ax.pie(counts, labels=labels, autopct="%.1f%%", explode=[
               0.01]*2, pctdistance=0.5, colors=[colors[1], colors[-2]], startangle=90)
        ax.set_title(f"{age}")
    

    plt.tight_layout()
