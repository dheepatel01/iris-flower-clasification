"""
Iris Flower Classification
==========================
A complete machine learning pipeline to classify Iris flower species
(Setosa, Versicolor, Virginica) using sepal and petal measurements.

Dataset: Iris.csv (150 samples, 3 classes, 4 features)
Models  : Logistic Regression, K-Nearest Neighbors, Decision Tree,
          Random Forest, Support Vector Machine
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette used throughout
PALETTE = {"Iris-setosa": "#4C72B0", "Iris-versicolor": "#55A868", "Iris-virginica": "#C44E52"}

# =============================================================================
# 1. LOAD & EXPLORE DATA
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(columns=["Id"], inplace=True)           # Id is not a feature
    return df


def explore_data(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("  IRIS FLOWER CLASSIFICATION — DATA EXPLORATION")
    print("=" * 60)

    print(f"\n📊 Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📋 Columns : {list(df.columns)}")

    print("\n── First 5 rows ──")
    print(df.head().to_string(index=False))

    print("\n── Data Types ──")
    print(df.dtypes.to_string())

    print("\n── Missing Values ──")
    print(df.isnull().sum().to_string())

    print("\n── Class Distribution ──")
    counts = df["Species"].value_counts()
    for species, n in counts.items():
        print(f"  {species:20s}: {n:3d} samples  ({n/len(df)*100:.1f}%)")

    print("\n── Statistical Summary ──")
    print(df.describe().round(3).to_string())


# =============================================================================
# 2. VISUALISATIONS
# =============================================================================

def plot_class_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")

    counts = df["Species"].value_counts()
    colors = [PALETTE[s] for s in counts.index]

    # Bar chart
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
    axes[0].set_title("Sample Count per Species")
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, 65)
    for i, (species, n) in enumerate(counts.items()):
        axes[0].text(i, n + 0.5, str(n), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, colors=colors,
                autopct="%1.0f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Proportion per Species")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved: 01_class_distribution.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Feature Distributions by Species", fontsize=14, fontweight="bold")

    for col, (ax_hist, ax_box) in zip(features, axes.T):
        # Histogram / KDE
        for species, grp in df.groupby("Species"):
            ax_hist.hist(grp[col], bins=12, alpha=0.6,
                         color=PALETTE[species], label=species, edgecolor="white")
        ax_hist.set_title(col.replace("Cm", " (cm)"))
        ax_hist.set_xlabel("cm")
        ax_hist.legend(fontsize=7)

        # Box plot
        data_by_species = [df[df["Species"] == s][col].values for s in PALETTE]
        bp = ax_box.boxplot(data_by_species, patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], PALETTE.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax_box.set_xticklabels(["Setosa", "Versicol.", "Virginica"], fontsize=8)
        ax_box.set_title(f"{col.replace('Cm','')} — Boxplot")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_feature_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved: 02_feature_distributions.png")


def plot_pairplot(df: pd.DataFrame) -> None:
    g = sns.pairplot(df, hue="Species", palette=PALETTE,
                     diag_kind="kde", plot_kws=dict(alpha=0.7, edgecolor="none"),
                     height=2.2)
    g.fig.suptitle("Pairplot of All Features", y=1.02, fontsize=14, fontweight="bold")
    g.fig.savefig(os.path.join(OUTPUT_DIR, "03_pairplot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved: 03_pairplot.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df.drop(columns="Species").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=mask, ax=ax, linewidths=0.5,
                annot_kws={"size": 11})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved: 04_correlation_heatmap.png")


def plot_scatter_petal(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Key Scatter Plots", fontsize=14, fontweight="bold")

    pairs = [
        ("PetalLengthCm", "PetalWidthCm"),
        ("SepalLengthCm", "SepalWidthCm"),
    ]
    titles = ["Petal Length vs Petal Width", "Sepal Length vs Sepal Width"]

    for ax, (x, y), title in zip(axes, pairs, titles):
        for species, grp in df.groupby("Species"):
            ax.scatter(grp[x], grp[y], c=PALETTE[species],
                       label=species, alpha=0.75, edgecolors="white", linewidth=0.4, s=55)
        ax.set_xlabel(x.replace("Cm", " (cm)"))
        ax.set_ylabel(y.replace("Cm", " (cm)"))
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_scatter_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved: 05_scatter_plots.png")


# =============================================================================
# 3. PREPROCESSING
# =============================================================================

def preprocess(df: pd.DataFrame):
    X = df.drop(columns="Species").values
    y = df["Species"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)          # setosa=0, versicolor=1, virginica=2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"\n  Train samples : {len(X_train)}  |  Test samples : {len(X_test)}")
    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, le, scaler


# =============================================================================
# 4. MODEL TRAINING & EVALUATION
# =============================================================================

def build_models():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=200, random_state=42), True),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=5), True),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=4, random_state=42), False),
        "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), False),
        "Support Vector Machine": (SVC(kernel="rbf", C=1.0, probability=True, random_state=42), True),
    }


def train_and_evaluate(models, X_train, X_test, X_train_sc, X_test_sc,
                        y_train, y_test, le) -> pd.DataFrame:
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    class_names = le.classes_

    print("\n" + "=" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 60)

    trained_models = {}

    for name, (model, needs_scaling) in models.items():
        Xtr = X_train_sc if needs_scaling else X_train
        Xte = X_test_sc  if needs_scaling else X_test

        # Fit
        model.fit(Xtr, y_train)
        trained_models[name] = (model, needs_scaling)

        # Predictions
        y_pred = model.predict(Xte)
        test_acc = accuracy_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")

        results.append({
            "Model":           name,
            "Test Accuracy":   round(test_acc * 100, 2),
            "CV Mean Acc (%)": round(cv_scores.mean() * 100, 2),
            "CV Std (%)":      round(cv_scores.std() * 100, 2),
        })

        print(f"\n{'─'*40}")
        print(f"  {name}")
        print(f"  Test Accuracy : {test_acc*100:.2f}%")
        print(f"  CV Accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

    return pd.DataFrame(results), trained_models


# =============================================================================
# 5. RESULT VISUALISATIONS
# =============================================================================

def plot_model_comparison(results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    w = 0.35

    bars1 = ax.bar(x - w/2, results_df["Test Accuracy"],   width=w, label="Test Accuracy",
                   color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + w/2, results_df["CV Mean Acc (%)"], width=w, label="CV Mean Accuracy",
                   color="#55A868", edgecolor="white",
                   yerr=results_df["CV Std (%)"], capsize=4, error_kw=dict(ecolor="black"))

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(85, 105)
    ax.set_title("Model Comparison — Test vs Cross-Validation Accuracy",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved: 06_model_comparison.png")


def plot_confusion_matrices(trained_models, X_test, X_test_sc, y_test, le) -> None:
    class_names = le.classes_
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle("Confusion Matrices (Test Set)", fontsize=13, fontweight="bold")

    for ax, (name, (model, needs_scaling)) in zip(axes, trained_models.items()):
        Xte = X_test_sc if needs_scaling else X_test
        y_pred = model.predict(Xte)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Setosa", "Versicol.", "Virginica"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved: 07_confusion_matrices.png")


def plot_feature_importance(trained_models, feature_names) -> None:
    rf_model = trained_models["Random Forest"][0]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i].replace("Cm", " (cm)") for i in indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    ax.barh(sorted_features[::-1], importances[indices][::-1],
            color=[colors[i] for i in indices[::-1]], edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for i, (feat, val) in enumerate(zip(sorted_features[::-1], importances[indices][::-1])):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved: 08_feature_importance.png")


def plot_decision_tree(trained_models, feature_names, le) -> None:
    dt_model = trained_models["Decision Tree"][0]
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(dt_model,
              feature_names=[f.replace("Cm", "") for f in feature_names],
              class_names=["Setosa", "Versicol.", "Virginica"],
              filled=True, rounded=True, fontsize=10, ax=ax,
              impurity=True, proportion=False)
    ax.set_title("Decision Tree (max_depth=4)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_decision_tree.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved: 09_decision_tree.png")


def plot_knn_boundary(X_train_sc, y_train, le) -> None:
    """Plot 2-D decision boundary using the two most important features (petal dims)."""
    # Features 2,3 → PetalLengthCm, PetalWidthCm (scaled indices)
    X2 = X_train_sc[:, 2:4]
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X2, y_train)

    h = 0.02
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.25,
                colors=list(PALETTE.values()), levels=[-0.5, 0.5, 1.5, 2.5])
    species_list = list(PALETTE.keys())
    for cls_idx, (species, color) in enumerate(PALETTE.items()):
        mask = y_train == cls_idx
        ax.scatter(X2[mask, 0], X2[mask, 1], c=color, label=species,
                   edgecolors="white", linewidth=0.4, s=60, alpha=0.9)
    ax.set_xlabel("Petal Length (scaled)")
    ax.set_ylabel("Petal Width (scaled)")
    ax.set_title("KNN Decision Boundary (Petal Dims, k=5)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_knn_decision_boundary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔  Saved: 10_knn_decision_boundary.png")


# =============================================================================
# 6. SUMMARY REPORT
# =============================================================================

def print_summary(results_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(results_df.sort_values("Test Accuracy", ascending=False).to_string(index=False))
    best = results_df.loc[results_df["Test Accuracy"].idxmax()]
    print(f"\n🏆  Best model  : {best['Model']}")
    print(f"    Test Acc   : {best['Test Accuracy']}%")
    print(f"    CV Acc     : {best['CV Mean Acc (%)']}% ± {best['CV Std (%)']}%")
    print(f"\n📁  All plots saved to: {OUTPUT_DIR}/")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    data_path = os.path.join(os.path.dirname(__file__), "Iris.csv")

    # 1. Load & explore
    df = load_data(data_path)
    explore_data(df)

    # 2. Visualise raw data
    print("\n📈 Generating exploratory plots …")
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_pairplot(df)
    plot_correlation_heatmap(df)
    plot_scatter_petal(df)

    # 3. Preprocess
    print("\n⚙️  Preprocessing …")
    (X_train, X_test,
     X_train_sc, X_test_sc,
     y_train, y_test, le, scaler) = preprocess(df)

    # 4. Train & evaluate
    models = build_models()
    results_df, trained_models = train_and_evaluate(
        models, X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, le
    )

    # 5. Result plots
    print("\n📊 Generating result plots …")
    plot_model_comparison(results_df)
    plot_confusion_matrices(trained_models, X_test, X_test_sc, y_test, le)
    feature_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    plot_feature_importance(trained_models, feature_names)
    plot_decision_tree(trained_models, feature_names, le)
    plot_knn_boundary(X_train_sc, y_train, le)

    # 6. Summary
    print_summary(results_df)


if __name__ == "__main__":
    main()
