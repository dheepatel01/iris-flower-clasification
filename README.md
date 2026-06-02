# 🌸 Iris Flower Classification

A complete machine learning project to classify Iris flower species — **Setosa**, **Versicolor**, and **Virginica** — based on sepal and petal measurements.

---

## 📌 Project Overview

The Iris dataset is one of the most well-known datasets in machine learning. This project covers the full ML pipeline:

- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Scaling
- Training 5 different classification models
- Evaluating models using accuracy, precision, recall, F1-score
- 5-Fold Cross-Validation
- Visualizations: pairplots, confusion matrices, decision boundaries, feature importance

---

## 📁 Project Structure

```
iris-flower-classification/
│
├── Iris.csv                     # Dataset (150 samples, 4 features, 3 classes)
├── iris_classification.py       # Main ML pipeline script
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── output/                      # Generated plots (auto-created on run)
    ├── 01_class_distribution.png
    ├── 02_feature_distributions.png
    ├── 03_pairplot.png
    ├── 04_correlation_heatmap.png
    ├── 05_scatter_plots.png
    ├── 06_model_comparison.png
    ├── 07_confusion_matrices.png
    ├── 08_feature_importance.png
    ├── 09_decision_tree.png
    └── 10_knn_decision_boundary.png
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Samples | 150 |
| Features | 4 (SepalLength, SepalWidth, PetalLength, PetalWidth) |
| Classes | 3 (Setosa, Versicolor, Virginica) |
| Missing Values | None |
| Class Balance | Perfectly balanced (50 samples each) |

---

## 🤖 Models Used

| Model | Test Accuracy | CV Accuracy |
|---|---|---|
| **Support Vector Machine (RBF)** | **96.67%** | **96.67% ± 1.67%** |
| Logistic Regression | 93.33% | 95.83% ± 2.64% |
| K-Nearest Neighbors (k=5) | 93.33% | 95.83% ± 2.64% |
| Decision Tree (max_depth=4) | 93.33% | 94.17% ± 2.04% |
| Random Forest (100 trees) | 90.00% | 95.00% ± 3.12% |

> 🏆 **Best Model: Support Vector Machine** with 96.67% test accuracy and lowest variance.

---

## 📈 Key Findings

- **Iris-setosa** is perfectly separable from the other two species using petal dimensions alone
- **Petal Length** and **Petal Width** are the most important features (Random Forest importance)
- **Iris-versicolor** and **Iris-virginica** have slight overlap in feature space, making them harder to separate
- SVM with RBF kernel handles the non-linear boundary best

---

## 🖼️ Sample Visualizations

The script automatically generates 10 plots saved to the `output/` folder:

1. Class Distribution (Bar + Pie)
2. Feature Distributions (Histograms + Box plots)
3. Pairplot of all features
4. Correlation Heatmap
5. Scatter Plots (Petal & Sepal dimensions)
6. Model Accuracy Comparison
7. Confusion Matrices (all 5 models)
8. Random Forest Feature Importance
9. Decision Tree Diagram
10. KNN Decision Boundary

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/iris-flower-classification.git
cd iris-flower-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python iris_classification.py
```

All plots will be saved to the `output/` folder automatically.

---

## 🛠️ Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## 📚 Concepts Covered

- Supervised Learning (Multi-class Classification)
- Train/Test Split & Stratified Sampling
- Feature Scaling (StandardScaler)
- Model Evaluation: Accuracy, Precision, Recall, F1-Score
- Cross-Validation (StratifiedKFold)
- Confusion Matrix
- Feature Importance
- Decision Boundary Visualization

---

## 👤 Author

**Dheeraj**  
Feel free to ⭐ this repo if you found it helpful!

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
