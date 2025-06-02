# FinalTask_IDX_Partners_DataScientist_MuhammadDzaky
# 💳 Credit Risk Prediction with Machine Learning

This repository contains a complete end-to-end data science project developed during the internship final task at ID/X Partners. The goal is to predict loan default risks using historical Lending Club data from 2007 to 2014.

The project uses a business-focused approach based on the CRISP-DM methodology, covering data understanding, feature engineering, exploratory data analysis, data preparation, modeling, evaluation, and business recommendations.

---

The dataset consists of 466,284 loan records and 75 features related to borrower profiles, loan amounts, payment statuses, and credit history. The main classification target is the `loan_status` feature, which is simplified into two categories:
- `GOOD`: Fully Paid
- `BAD`: Charged Off, Late, Default, and other failed statuses

Because the dataset is heavily imbalanced (GOOD >> BAD), the modeling and evaluation strategies were carefully designed to ensure meaningful risk detection.

---

Key steps in this project include:

- **Data Understanding**  
  Identified imbalance in label distribution and a need to consolidate loan statuses into binary labels (GOOD vs BAD).

- **Feature Engineering**  
  Removed 12 non-informative columns (e.g., ID, URL, desc), defined target label, and inspected missing values.

- **Exploratory Data Analysis (EDA)**  
  Conducted label distribution visualization and correlation heatmap to observe feature relationships and redundancy.

- **Data Preparation**  
  Applied one-hot encoding for categorical features, normalized numeric features using StandardScaler, label-encoded the target, and performed stratified 80:20 train-test split.

- **Modeling**  
  Implemented two classification models: Logistic Regression and Random Forest. Logistic Regression serves as a baseline model, while Random Forest provides stronger recall and feature flexibility.

- **Evaluation**  
  Models were evaluated using Accuracy, ROC AUC, Precision, Recall, and F1-Score. Results show Random Forest outperforms the baseline:
  Visual outputs include countplots, feature importance charts, and predicted probability distributions.

---

### Business Impact & Recommendations

The model supports data-driven credit scoring by identifying high-risk applicants before approval. This can reduce financial losses and improve loan portfolio quality. Future enhancements include:

- Integrating third-party credit scoring data
- Balancing the dataset with techniques like SMOTE
- Performing periodic model retraining (e.g., quarterly)
- Deploying the model through APIs or dashboard scoring systems

---

### How to Run the Project

**Requirements**:
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

**Setup**:
1. Clone the repository
2. Install dependencies  
```bash
pip install -r requirements.txt
