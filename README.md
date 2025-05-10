# Cirrhosis Data Analysis and Synthetic Data Generation

## Project Overview
This project analyzes a cirrhosis dataset to explore the distribution and relationships of key clinical features (Bilirubin, Albumin, SGOT, Prothrombin) across different disease stages. It includes preprocessing, exploratory data analysis (EDA), statistical testing, synthetic data generation, and machine learning model evaluation to assess synthetic data utility. The analysis is conducted in Jupyter Notebook (.ipynb) format using Python with libraries such as Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, PyTorch, and LightGBM.

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Preprocessing](#preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Synthetic Data Generation](#synthetic-data-generation)
7. [Statistical Analysis](#statistical-analysis)
8. [Machine Learning Models](#machine-learning-models)
9. [Results](#results)
10. [Usage](#usage)
11. [Contributing](#contributing)
12. [License](#license)

## Installation
To run this project, ensure Python 3.8+ is installed. Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/cirrhosis-analysis.git
cd cirrhosis-analysis
pip install -r requirements.txt
```

## Requirements
The requirements.txt file includes:
1. pandas
2. numpy
3. matplotlib
4. seaborn
5. scipy
6. scikit-learn
7. tensorflow
8. torch
9. lightgbm
10. statsmodels

## Dataset
The dataset (cirrhosis.csv) contains clinical measurements for patients with cirrhosis, including:

- Bilirubin: Blood pigment levels (mg/dL)
- Albumin: Protein levels in blood (g/dL)
- SGOT: Liver enzyme levels (U/mL)
- Prothrombin: Blood clotting time (seconds)
- Stage: Disease stage (1 to 4)

Synthetic data is generated and saved as synthetic_cirrhosis_data2.csv for comparison with the real data.
## Project Structure
cirrhosis-analysis/
│
├── data/
│   ├── cirrhosis.csv
│   ├── preprocessed_cirrhosis2.csv
│   ├── synthetic_cirrhosis_data2.csv
│
├── scripts/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── synthetic_data_generation.py
│   ├── statistical_analysis.py
│   ├── ml_models.py
│
├── figures/
│   ├── stage_distribution.png
│   ├── histograms.png
│   ├── box_plots.png
│   ├── pair_plot.png
│   ├── scatter_bilirubin_albumin.png
│   ├── 3d_scatter.png
│
├── requirements.txt
├── README.md

Preprocessing
The preprocessing steps include:

Handling Missing Values: Numerical columns (Bilirubin, Albumin, SGOT, Prothrombin) are filled with their median values.
Log Transformation: Applied to skewed features (Bilirubin, SGOT) to reduce skewness.
Feature Selection: Selected relevant columns for analysis.
Interaction Terms: Added features like Bilirubin_SGOT and Albumin_Prothrombin for machine learning.
Scaling: Features are standardized using StandardScaler for machine learning models.

Run the preprocessing script:
python scripts/preprocessing.py

Exploratory Data Analysis (EDA)
EDA is performed to visualize and understand the data distribution and relationships:

Histograms: Show distributions of features (original and log-transformed).
Box Plots: Display feature distributions stratified by stage.
Pair Plots: Visualize pairwise relationships between key features.
Scatter Plots: Highlight relationships like Bilirubin vs. Albumin.
3D Scatter Plot: Visualizes Bilirubin, Albumin, and SGOT by stage.
Stage Distribution: Bar plot to check for class imbalance.

Run the EDA script:
python scripts/eda.py

Synthetic Data Generation
Synthetic data is generated to mimic the real dataset's statistical properties. The process includes:

Using a generative model (likely GAN-based, as implied by the code).
Evaluating synthetic data quality using Kolmogorov-Smirnov (KS) tests across epochs.
Saving synthetic data to synthetic_cirrhosis_data2.csv.

Run the synthetic data generation script:
python scripts/synthetic_data_generation.py

Statistical Analysis
Statistical tests compare real and synthetic data:

T-Tests: Compare means of continuous features.
Cohen's d: Measure effect size for mean differences.
KS-Tests: Compare feature distributions.
Chi-Square Test: Compare stage distributions.
Correlation Analysis: Compare correlation matrices between real and synthetic data.

Run the statistical analysis script:
python scripts/statistical_analysis.py

Key Findings

Bilirubin and Albumin means are significantly different (p < 0.05).
SGOT and Prothrombin means are not significantly different (p > 0.05).
Stage distributions are significantly different (p < 0.05).
Matching Correlation Coefficient: Average absolute difference of 0.0600.

Machine Learning Models
The project evaluates multiple models for classifying cirrhosis stages (binary classification: stages 1-2 vs. 3-4):

Models Evaluated:
Gradient Boosting
Random Forest
Logistic Regression
LightGBM
Multi-Layer Perceptron (MLP)
Voting Classifier (Logistic Regression + LightGBM)


Feature Selection: SelectKBest with f_classif to select top 4 features.
Hyperparameter Tuning: GridSearchCV for optimizing model parameters.
Cross-Validation: 5-fold stratified cross-validation.
Metrics: Accuracy, Precision, Recall, Specificity, F1-Score, AUC-ROC.

Run the machine learning script:
python scripts/ml_models.py

Model Performance



Model
Accuracy
Precision
Recall
Specificity
F1-Score
AUC-ROC



Gradient Boosting
0.7112
0.7375
0.9365
0.1166
0.8246
0.6034


Random Forest
0.7257
0.7546
0.9232
0.2047
0.8299
0.6202


Logistic Regression
0.7354
0.7669
0.9132
0.2656
0.8335
0.6567


LightGBM
0.7257
0.7268
0.9967
0.0087
0.8406
0.6502


MLP
0.7112
0.7348
0.9433
0.0984
0.8257
0.6164


Voting Classifier
0.7257
0.7257
1.0000
0.0000
0.8411
0.6587



ANOVA Results: No significant difference in accuracy between models (p = 0.1938).

Results

EDA: Revealed significant skewness in Bilirubin and SGOT, addressed via log transformation. Stage distribution showed slight imbalance.
Synthetic Data: Closely mimics real data but differs significantly in Bilirubin, Albumin, and Stage distributions.
Machine Learning: Logistic Regression achieved the highest accuracy (0.7354), but synthetic data training yielded lower accuracy (0.5542) compared to real data (0.7108).
Utility: Synthetic data is useful but not a perfect substitute for real data in machine learning tasks.

Usage
To reproduce the analysis:

Ensure all dependencies are installed.
Place cirrhosis.csv in the data/ directory.
Run scripts in the following order:python scripts/preprocessing.py
python scripts/eda.py
python scripts/synthetic_data_generation.py
python scripts/statistical_analysis.py
python scripts/ml_models.py


Outputs (figures, synthetic data, and results) will be saved in figures/ and data/ directories.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
