import os
import time
from dataclasses import dataclass
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

from hyperfreedt import build_tree, count_leaves, predict


@dataclass
class TestResult:
    score: float
    leaf_count: int


@dataclass
class TestSummary:
    dataset_name: str
    mean_msi_score: float
    mean_msi_leaf_count: float
    mean_dtc_score: float
    mean_dtc_leaf_count: float


def preprocess_data(dataset):
    x = dataset.data.features
    for colname in list(x.columns):
        if x[colname].dtype is np.dtype(object):
            x.loc[:, f"encoded_{colname}"] = OrdinalEncoder(
                encoded_missing_value=-1
            ).fit_transform(x[[colname]])
            x.pop(colname)
        elif x[colname].isnull().sum():
            x.loc[:, f"fillna_{colname}"] = x[colname].fillna(x[colname].median())
            x.pop(colname)
    y = dataset.data.targets.values.ravel()
    if y.dtype is np.dtype(object):
        y = OrdinalEncoder().fit_transform(y.reshape(-1, 1)).ravel()
    x["__target__"] = y
    return x


def fetch_data(dataset_name):
    filename = f"{dataset_name}.csv"
    if os.path.isfile(filename):
        x = pd.read_csv(filename)
    else:
        dataset = fetch_ucirepo(name=dataset_name)
        x = preprocess_data(dataset)
        x.to_csv(filename, index=False)
    y = x.pop("__target__").values
    return x.values, y


def test_data_msi(x, y, rs):
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.3, random_state=rs
    )
    tree = build_tree(train_x, train_y)
    pred = predict(tree, test_x)
    return TestResult(np.mean(np.argmax(pred, axis=1) == test_y), count_leaves(tree))


def test_data_dtc(x, y, rs, min_samples_leaf=5, max_leaf_nodes=99999):
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.3, random_state=rs
    )
    dtc = DecisionTreeClassifier(
        min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes
    )
    dtc.fit(train_x, train_y)
    pred = dtc.predict(test_x)
    return TestResult(np.mean(pred == test_y), dtc.get_n_leaves())


def test_data_loop(x, y, ds_name):
    with Pool(24) as pool:
        arg_gen = ((x, y, rs) for rs in range(100))
        msi_results = pool.starmap(test_data_msi, arg_gen)
        arg_gen = ((x, y, rs, 5) for rs in range(100))
        dtc_results = pool.starmap(test_data_dtc, arg_gen)

    mean_msi_score = np.mean([result.score for result in msi_results])
    mean_dtc_score = np.mean([result.score for result in dtc_results])
    mean_msi_leaf_count = np.mean([result.leaf_count for result in msi_results])
    mean_dtc_leaf_count = np.mean([result.leaf_count for result in dtc_results])
    return TestSummary(
        ds_name,
        mean_msi_score,
        mean_msi_leaf_count,
        mean_dtc_score,
        mean_dtc_leaf_count,
    )


def create_bar_chart(summaries):
    # Extract data from the summaries list
    names = [s.dataset_name for s in summaries]
    msi_scores = [s.mean_msi_score for s in summaries]
    dtc_scores = [s.mean_dtc_score for s in summaries]
    msi_leaf_count = [s.mean_msi_leaf_count for s in summaries]
    dtc_leaf_count = [s.mean_dtc_leaf_count for s in summaries]

    # Set the width of the bars and the positions
    num_datasets = len(names)
    x = np.arange(num_datasets)  # x-coordinates for the groups
    bar_width = 0.35  # width of each bar

    # Create a figure with two subplots (one for each chart)
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # --- First chart: mean_msi_score vs mean_dtc_score ---
    ax1.bar(x - bar_width / 2, msi_scores, bar_width, label="MSI Score")
    ax1.bar(x + bar_width / 2, dtc_scores, bar_width, label="DTC Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("Score")
    ax1.set_title("Mean MSI vs DTC (MSL5) Scores by Dataset")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # --- Second chart: mean_msi_leaf_count vs mean_dtc_leaf_count ---
    ax2.bar(x - bar_width / 2, msi_leaf_count, bar_width, label="MSI Leaf Count")
    ax2.bar(x + bar_width / 2, dtc_leaf_count, bar_width, label="DTC Leaf Count")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("Leaf Count")
    ax2.set_title("Mean Leaf Count by Dataset")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Improve layout to prevent overlap
    plt.tight_layout()
    plt.savefig("chart.png")


def run():
    results = []
    for short_name, full_name in (
        ("Cancer", "Breast Cancer"),
        ("Digits", "Optical Recognition of Handwritten Digits"),
        ("Yeast", "Yeast"),
        ("Shuttle", "Statlog (Shuttle)"),
        ("Page", "Page Blocks Classification"),
        ("Image", "Image Segmentation"),
        ("Spam", "Spambase"),
        ("Landsat", "Statlog (Landsat Satellite)"),
        ("Magic", "MAGIC Gamma Telescope"),
        ("Wine", "Wine Quality"),
        ("Abalone", "Abalone"),
    ):
        start_time = time.time()
        x, y = fetch_data(full_name)
        test_summary = test_data_loop(x, y, short_name)
        results.append(test_summary)
        create_bar_chart(results)
        print(f"{time.time() - start_time} {short_name}", flush=True)


if __name__ == "__main__":
    run()
