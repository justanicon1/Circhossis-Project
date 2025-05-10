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
├── circhossis_v2104.ipynb
├── generator_stage_1.pth
├── generator_stage_2.pth
├── generator_stage_3.pth
├── generator_stage_4.pth
├── circhossis.csv
├── synthetic_circhossis_data2.csv
├── requirements.txt
├── README.md

## Preprocessing
The preprocessing steps include:

- Handling Missing Values: Numerical columns (Bilirubin, Albumin, SGOT, Prothrombin) are filled with their median values.
- Log Transformation: Applied to skewed features (Bilirubin, SGOT) to reduce skewness.
- Feature Selection: Selected relevant columns for analysis.
- Interaction Terms: Added features like Bilirubin_SGOT and Albumin_Prothrombin for machine learning.
- Scaling: Features are standardized using StandardScaler for machine learning models.

## Run the preprocessing script:
circhossis_v2104.ipynb

## Exploratory Data Analysis (EDA)
EDA is performed to visualize and understand the data distribution and relationships:

**Histograms:** Show distributions of features.
**Box Plots:** Display feature distributions stratified by stage.
**Pair Plots:** Visualize pairwise relationships between key features.
**Scatter Plots:** Highlight relationships like Bilirubin vs. Albumin.
**3D Scatter Plot:** Visualizes Bilirubin, Albumin, and SGOT by stage.
**Stage Distribution:** Bar plot to check for class imbalance.



## Synthetic Data Generation
Synthetic data is generated to mimic the real dataset's statistical properties. The process includes:

- Using a generative model(GAN).
- Evaluating synthetic data quality using Kolmogorov-Smirnov (KS) tests across epochs.
- Saving synthetic data to synthetic_cirrhosis_data2.csv.


## Statistical Analysis
Statistical tests compare real and synthetic data:

- **T-Tests:** Compare means of continuous features.
- **Cohen's d:** Measure effect size for mean differences.
- **KS-Tests:** Compare feature distributions.
- **Chi-Square Test:** Compare stage distributions.
- **Correlation Analysis:** Compare correlation matrices between real and synthetic data.


## Key Findings

- Bilirubin and Albumin means are significantly different (p < 0.05).
- SGOT and Prothrombin means are not significantly different (p > 0.05).
- Stage distributions are significantly different (p < 0.05).
- Matching Correlation Coefficient: Average absolute difference of 0.0600.

## Machine Learning Models
The project evaluates multiple models for classifying cirrhosis stages (binary classification: stages 1-2 vs. 3-4):

**Models Evaluated:**
1. Gradient Boosting
2. Random Forest
3. Logistic Regression
4. LightGBM


## Feature Selection: SelectKBest with f_classif to select top 4 features.
- Hyperparameter Tuning: GridSearchCV for optimizing model parameters.
- Cross-Validation: 5-fold stratified cross-validation.
- Metrics: Accuracy, Precision, Recall, Specificity, F1-Score, AUC-ROC.

## Model Performance

| Model              | Accuracy | Precision | Recall  | Specificity | F1-Score | AUC-ROC |
|--------------------|----------|-----------|---------|-------------|----------|---------|
| Gradient Boosting  | 0.7112   | 0.7375    | 0.9365  | 0.1166      | 0.8246   | 0.6034  |
| Random Forest      | 0.7257   | 0.7546    | 0.9232  | 0.2047      | 0.8299   | 0.6202  |
| Logistic Regression| 0.7354   | 0.7669    | 0.9132  | 0.2656      | 0.8335   | 0.6567  |
| LightGBM           | 0.7257   | 0.7268    | 0.9967  | 0.0087      | 0.8406   | 0.6502  |
| MLP                | 0.7112   | 0.7348    | 0.9433  | 0.0984      | 0.8257   | 0.6164  |
| Voting Classifier  | 0.7257   | 0.7257    | 1.0000  | 0.0000      | 0.8411   | 0.6587  |




**ANOVA Results:** No significant difference in accuracy between models (p = 0.1938).

## Results

1. EDA: Revealed significant skewness in Bilirubin and SGOT, addressed via log transformation. Stage distribution showed slight imbalance.
2. Synthetic Data: Closely mimics real data but differs significantly in Bilirubin, Albumin, and Stage distributions.
3. Machine Learning: Logistic Regression achieved the highest accuracy (0.7354), but synthetic data training yielded lower accuracy (0.5542) compared to real data (0.7108).

## Usage
To reproduce the analysis:

Ensure all dependencies are installed.
Place cirrhosis.csv in the data/ directory.
Run the circhossis_v2104.ipynb script


Outputs (figures, synthetic data, and results) will be saved in the project's directory

## Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
