
# Airplane Crash Severity Prediction using Machine Learning

## Project Overview
This project analyzes historical airplane crash data to predict whether a crash is **Severe** or **Non-Severe**.  
The dataset comes from [Airplane Crashes and Fatalities Since 1908 (Kaggle)](https://www.kaggle.com/datasets/saurograndi/airplane-crashes-since-1908).  

By applying exploratory data analysis (EDA), preprocessing, class imbalance handling, and machine learning models, this project demonstrates how predictive analytics can provide insights into aviation safety.

## Methodology
1. **Data Preprocessing**
   - Selected features: `Operator`, `Aircraft Type`, `Country`, `Passenger Count`
   - Handled missing values and encoded categorical variables (One-Hot Encoding)
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance classes

2. **Exploratory Data Analysis**
   - Severe crashes dominate the dataset
   - Higher passenger counts correlated with severity
   - Certain operators (e.g., Military, Pan Am, Aeroflot) and older aircraft types (Douglas DC series, Yakovlev YAK-40) were linked to higher severity

3. **Models Trained**
   - Logistic Regression  
   - Random Forest  
   - Gradient Boosting  
   - Support Vector Machine (SVM)  
   - K-Nearest Neighbors (KNN)  
   - XGBoost  
   - LightGBM  

4. **Evaluation Metrics**
   - Accuracy  
   - Macro F1-Score (important for imbalanced data)  
   - ROC AUC  

## Results

| Model                | Accuracy | Macro F1 | ROC AUC |
|-----------------------|----------|----------|---------|
| Logistic Regression   | 0.957    | 0.441    | 0.671   |
| Random Forest         | 0.978    | 0.529    | 0.694   |
| Gradient Boosting     | 0.981    | 0.585    | **0.727** |
| SVM                   | 0.969    | 0.508    | 0.692   |
| KNN                   | 0.945    | 0.442    | 0.667   |
| XGBoost               | 0.983    | 0.574    | 0.701   |
| LightGBM              | **0.985** | **0.596** | 0.706   |

**Key Findings**
- Boosting models (Gradient Boosting, LightGBM, XGBoost) clearly outperformed simpler models  
- LightGBM offered the best overall balance of Accuracy and F1  
- Gradient Boosting achieved the highest ROC AUC, making it most reliable for ranking Severe vs Non-Severe cases  
- Passenger count and aircraft/operator type were the most critical features  

## How to Run This Project

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
````

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Jupyter Notebook

```bash
jupyter notebook
```

Open **Airplane Crash Severity Prediction using Machine Learning.ipynb** and run all cells.

## Project Structure

```
├── Airplane Crash Severity Prediction using Machine Learning.ipynb
├── requirements.txt
├── README.md
└── data/
    └── Airplane_Crashes_and_Fatalities_Since_1908.csv
```

## Future Improvements

* Add more aviation datasets for better generalization
* Try deep learning (e.g., LSTMs on crash narratives)
* Deploy a web dashboard for real-time predictions

## Tools & Libraries

* Python: Pandas, NumPy, Matplotlib, Seaborn
* Machine Learning: Scikit-learn, Imbalanced-learn (SMOTE), XGBoost, LightGBM
* Visualization: Folium, WordCloud

## Acknowledgments

* Dataset: [Kaggle – Airplane Crashes and Fatalities Since 1908](https://www.kaggle.com/datasets/saurograndi/airplane-crashes-since-1908)
* Libraries: Scikit-learn, Imbalanced-learn, XGBoost, LightGBM, Matplotlib, Seaborn

```


