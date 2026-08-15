import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

BCS_COLUMN_NAME = 'bcs_mean'
MAIN_PATH = '../BCS_dataset'
image_dir = "../BCS_dataset/images"


def get_and_show_dataset():
    df = pd.read_csv(f"{MAIN_PATH}/labels.csv")
    print(f"number of total images: {len(df)}")

    if len(df) > 0:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        ax = sns.histplot(data=df, x=BCS_COLUMN_NAME, bins=20, kde=True, color='teal')

        plt.title(f'Distribution of Body Condition Score (BCS)\nTotal Samples: {len(df)}', fontsize=15)
        plt.xlabel('BCS (Exact Mean Values)', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)

        plt.xlim(2, 5)

        plt.show()

    return df


def analyze_dataset_corrupted_height_width(df):

    corrupted_files = []
    black_images = []
    white_images = []

    widths = []
    heights = []

    black_threshold = 5
    white_threshold = 250

    for filename in os.listdir(image_dir):
        path = os.path.join(image_dir, filename)

        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        try:
            img = Image.open(path)

            width, height = img.size
            widths.append(width)
            heights.append(height)

            img_array = np.array(img)

            mean_pixel = img_array.mean()

            if mean_pixel < black_threshold:
                black_images.append(filename)
            elif mean_pixel > white_threshold:
                white_images.append(filename)

        except:
            corrupted_files.append(filename)

    print(f"number of corrupted images: {len(corrupted_files)}")
    print(f"number of black images: {len(black_images)}")
    print(f"number of white images: {len(white_images)}")

    print(f"range of width of images: {list(set(widths))}")
    print(f"range of height of images: {list(set(heights))}")

    bad_files = set()
    bad_files.update(corrupted_files)
    bad_files.update(white_images)
    bad_files.update(black_images)
    df_clean = df[~df['image_filename'].isin(bad_files)].copy()

    # width distribution
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20)
    plt.title("Distribution of Image Width")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Number of Images")
    plt.grid(alpha=0.3)

    # height distribution
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20)
    plt.title("Distribution of Image Height")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Number of Images")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df_clean


def stratified_group_k_fold_splitting(df):
    df['cow_id'] = df['image_filename'].apply(lambda x: os.path.basename(x).split('_')[0])
    cow_counts = df['cow_id'].value_counts()
    print(f"number of different cows: {len(cow_counts)}")

    df['bcs_bin'] = pd.qcut(df[BCS_COLUMN_NAME], 4, labels=False, duplicates='drop')
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)  # n_splits=7 -> each fold: 14.3%

    X = df
    y = df['bcs_bin']
    groups = df['cow_id']

    folds = []

    # building 7 folds containing 1/7 of data for test and 6/7 of data for train
    # we have 7 different splits of total of data with different test part
    for train_idx, test_idx in sgkf.split(X, y, groups):
        folds.append((train_idx, test_idx))

    print(f"Number of folds: {len(folds)}")

    # choosing test and validation folds based on distribution
    target_distribution = (  # distribution of bcs in total dataset
        df['bcs_bin']
        .value_counts(normalize=True)  # giving each bcn_bin percentages
        .sort_index()
    )
    target_size = len(df)

    best_score = np.inf
    best_split = None

    for val_fold in range(len(folds)):

        for test_fold in range(len(folds)):

            if val_fold == test_fold:
                continue

            val_idx = folds[val_fold][1]  # folds[val_fold] = ((train_idx), (test_idx))
            test_idx = folds[test_fold][1]

            train_idx = np.setdiff1d(  # removing test and validation indexes from all indexes to make train indexes
                np.arange(len(df)),  # all possible indexes
                np.concatenate([val_idx, test_idx])  # merging test and validation indexes
            )

            # Split sizes
            train_ratio = len(train_idx) / target_size
            val_ratio = len(val_idx) / target_size
            test_ratio = len(test_idx) / target_size

            # Distribution of BCS bins
            train_distribution = (
                df.iloc[train_idx]['bcs_bin']
                .value_counts(normalize=True)  # percentage of each possible bcs(in 4 group of bcd_bin)
                .reindex(target_distribution.index, fill_value=0)
            )

            val_distribution = (
                df.iloc[val_idx]['bcs_bin']
                .value_counts(normalize=True)
                .reindex(target_distribution.index, fill_value=0)
            )

            test_distribution = (
                df.iloc[test_idx]['bcs_bin']
                .value_counts(normalize=True)
                .reindex(target_distribution.index, fill_value=0)
            )

            # Distribution error
            distribution_error = (
                    np.abs(train_distribution - target_distribution).sum()
                    +
                    np.abs(val_distribution - target_distribution).sum()
                    +
                    np.abs(test_distribution - target_distribution).sum()
            )

            # Size error
            size_error = (
                    abs(train_ratio - 0.70)
                    +
                    abs(val_ratio - 0.15)
                    +
                    abs(test_ratio - 0.15)
            )

            # Total score
            score = distribution_error + size_error

            if score < best_score:
                best_score = score

                best_split = {
                    'train_idx': train_idx,
                    'val_idx': val_idx,
                    'test_idx': test_idx,
                    'val_fold': val_fold,
                    'test_fold': test_fold
                }

    train_df = df.iloc[best_split['train_idx']].copy()
    test_df = df.iloc[best_split['test_idx']].copy()
    val_df = df.iloc[best_split['val_idx']].copy()

    train_df.drop(columns=['bcs_bin'], inplace=True)
    val_df.drop(columns=['bcs_bin'], inplace=True)
    test_df.drop(columns=['bcs_bin'], inplace=True)

    print(f"train ration: {len(train_df) / len(df)}")
    print(f"test ration: {len(test_df) / len(df)}")
    print(f"validation ration: {len(val_df) / len(df)}")

    # chck for leakage
    train_cows = set(train_df['cow_id'])
    test_cows = set(test_df['cow_id'])
    val_cows = set(val_df['cow_id'])

    print(f"train cows id: {train_cows}")
    print(f"test cows id: {test_cows}")
    print(f"validation cows id: {val_cows}")

    print(f"Train ∩ Validation: {train_cows & val_cows}")
    print(f"Train ∩ Test: {train_cows & test_cows}")
    print(f"Validation ∩ Test: {val_cows & test_cows}")

    # distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Train
    sns.histplot(train_df[BCS_COLUMN_NAME], bins=20, kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Train Distribution')
    axes[0].set_xlabel(BCS_COLUMN_NAME)
    axes[0].set_ylabel('Count')

    # Validation
    sns.histplot(val_df[BCS_COLUMN_NAME], bins=20, kde=True, ax=axes[1], color='green')
    axes[1].set_title('Validation Distribution')
    axes[1].set_xlabel(BCS_COLUMN_NAME)
    axes[1].set_ylabel('Count')

    # Test
    sns.histplot(test_df[BCS_COLUMN_NAME], bins=20, kde=True, ax=axes[2], color='red')
    axes[2].set_title('Test Distribution')
    axes[2].set_xlabel(BCS_COLUMN_NAME)
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    return train_df, test_df, val_df


if __name__ == '__main__':
    df = get_and_show_dataset()
    df = analyze_dataset_corrupted_height_width(df)
    train_df, test_df, val_df = stratified_group_k_fold_splitting(df)

    dataset_folder = MAIN_PATH
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    train_df.to_csv(os.path.join(dataset_folder, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_folder, 'test.csv'), index=False)
    val_df.to_csv(os.path.join(dataset_folder, 'val.csv'), index=False)

