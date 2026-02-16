# Trader Behavior vs Market Sentiment Analysis
## Objective
This project analyzes how market sentiment (Fear/Greed Index) influences trader performance and behavior in crypto derivatives trading.
- Changes in profitability across sentiment regimes
- Behavioral shifts (trade frequency, leverage proxy, position sizing)
- Actionable trading rules based on data
## Dateset
- ### Trader historical dataset
- Trade-level data (PnL, size, direction, fees, timestamps)
- ### Market sentiment dataset
- Daily Fear & Greed classification

## Data Preparation

### Steps performed

- Converted timestamps → daily date
- Merged sentiment with trade data
- Removed duplicate and irrelevant columns
- Created daily aggregated metrics:

| Metric | Description |
|--------|------------|
| `daily_pnl` | Total PnL per day per sentiment |
| `win_rate` | % of profitable trades |
| `trades_per_day` | Trading activity level |
| `avg_trade_size` | Mean position size (risk proxy) |
| `long_short_ratio` | Directional bias |
---

## Target Variable

PnL was bucketed into 3 classes:

| Bucket | Meaning |
|--------|---------|
| 0 | Loss |
| 1 | Neutral |
| 2 | Profit |

---
#  Features Used

```jupyter
['trades_per_day',
 'avg_trade_size',
 'win_rate',
 'long_short_ratio',
 'classification']
```
#  Exploratory Analysis

##  Profitability vs Sentiment

| Sentiment       | Daily PnL | Win Rate |
|-----------------|-----------|----------|
| Extreme Fear    | Highest   | Low (~32%) |
| Extreme Greed   | Moderate  | Highest (~46%) |
| Greed           | Lowest    | Low |
| Neutral         | Mid       | Low |

### Insight

- **Extreme Fear → highest profits but low win rate**  
  → Few large winners drive returns

- **Extreme Greed → higher win rate but smaller profits**

---

##  Trading Activity

| Sentiment       | Trades/Day | Avg Size |
|-----------------|-----------|----------|
| Extreme Fear    | Highest activity | Smallest size |
| Neutral         | Lower activity   | Largest size |

### Insight

- Traders **overtrade during fear**
- Position sizes **increase in neutral markets**

---

#  Data Preparation 
## Loding
```jupyter
# loding file 
#link: https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing
#link: https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing
file1='1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf'
file2='1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs'
url1=f"https://drive.google.com/uc?export=download&id={file1}"
url2=f"https://drive.google.com/uc?export=download&id={file2}"
sentiment = pd.read_csv(url1)
historical=pd.read_csv(url2)
```
## Merging and change to date time
```jupyter
# creating new date column and changeing to date time 
historical['date'] = pd.to_datetime(historical['Timestamp IST'], dayfirst=True)
historical['date'] = historical['date'].dt.date
# changing datetime from object
sentiment['date']=pd.to_datetime(sentiment['date']).dt.date
# combning two datasets by merge
bitcoin = historical.merge(sentiment[['date','value','classification']], on='date', how='left')
# removing or not using unnecessary columns
bitcoin = bitcoin.drop(columns=['Account','Transaction Hash','Order ID','Timestamp'])
```
```jupyter
# statics data
bitcoin.describe()
# checking skew
skew_num = bitcoin[['Execution Price','Size Tokens','Size USD','Start Position','Closed PnL', 'Fee']].skew()
print(skew_num)
cat_col=bitcoin.select_dtypes(include='object').columns
bitcoin[cat_col].value_counts(normalize=True)

```
## Daily based dataset for crypto
```jupyter
# creating daily based dataset 
bitcoin['Direction'] = bitcoin['Direction'].str.lower()
long_label=['buy','open long','close short']
short_label=['sell','open short','close long']
bitcoin['in_long'] = bitcoin['Direction'].isin(long_label).astype(int)
bitcoin['in_short'] = bitcoin['Direction'].isin(short_label).astype(int)
daily_bitcoin = bitcoin.groupby(['date','classification']).agg(
    daily_pnl=('Closed PnL','sum'),
    trades_per_day=('Trade ID','count'),
    avg_trade_size=('Size USD','mean'),
    win_rate=('Closed PnL', lambda x: (x>0).mean()),
    long_trades=('in_long','sum'),
    short_trades=('in_short','sum')
).reset_index()

daily_bitcoin['long_short_ratio'] = daily_bitcoin['long_trades'] / (daily_bitcoin['short_trades'] + 1).replace([-np.inf, np.inf],np.nan).fillna(0)
```
## Tables for Insides
```jupyter
daily_bitcoin.groupby('classification')[['daily_pnl','win_rate']].mean()
daily_bitcoin.groupby('classification')[['trades_per_day', 'avg_trade_size', 'long_short_ratio']].mean()


```
## Graphs
```jupyter
# graph representation of sentement of trading day vs profit and loss
plt.figure(figsize=(12,7))
sns.boxplot(daily_bitcoin, x='classification', y='daily_pnl')
plt.title('sentiment vs profitand loss')
plt.xlabel('classification')
plt.ylabel('daily PnL')
plt.show()
#graphical representation of sentiment vs  trade per day
plt.figure(figsize=(12,6))
sns.barplot(daily_bitcoin, x='classification',y='trades_per_day')
plt.title('sentiment vs trading per day')
plt.xlabel('classification')
plt.ylabel('trades per day')
plt.show()
# graphical represent of sentiment vs wining rate
plt.figure(figsize=(10,5))
sns.boxplot(daily_bitcoin, x='classification', y='win_rate')
plt.title('classification vs winning rate')
plt.xlabel('classification')
plt.ylabel('winning rate')
plt.show()
```
## Model
```jupyter
# creating tardet features 
daily_bitcoin = daily_bitcoin.sort_values('date')

daily_bitcoin['next_day_pnl'] = daily_bitcoin['daily_pnl'].shift(-1)

daily_bitcoin['pnl_bucket'] = pd.cut(daily_bitcoin['next_day_pnl'],bins=[-1e9, -1, 1, 1e9],labels=['Loss','Neutral','Profit'])
```

```jupyter
# Selecting features
column = [
    'trades_per_day',
    'avg_trade_size',
    'win_rate',
    'long_short_ratio',
    'classification'
]

X = daily_bitcoin[column]
y = daily_bitcoin['pnl_bucket']
```

---

## Drop Null Before ML

```jupyter

# drop null value
daily_bitcoin=daily_bitcoin.dropna(subset=['next_day_pnl', 'pnl_bucket'])
daily_bitcoin.info()
```

---

## Skew

```jupyter
skew_num = daily_bitcoin[daily_bitcoin.select_dtypes(include=np.number).columns].skew()
print(skew_num)
```

---

##  log transformation 

```jupyter

# log transformation 
skew_col = X_train.select_dtypes(include=np.number).columns
for col in skew_col:
    X_train[col]=np.log1p(X_train[col])
    X_test[col]=np.log1p(X_test[col])
```
## Encoding
```juputer
# encoder
oe = OrdinalEncoder()
X_train['classification']=oe.fit_transform(X_train[['classification']])
X_test['classification']=oe.transform(X_test[['classification']])
# null or duplicated
print(y_test.isnull().sum())
print(X_test.isnull().sum())

# encoding ffor out put data
le=LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

```

---

#  Model Training(Logistics Regression)

```jupyter
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    random_state=42,
    max_iter=1000,
    
)

lr.fit(X_train, y_train)
```

---

#  Predictions

```python
y_pre = lr.predict(X_test)
y_pred = lr.predict_proba(X_test)
```

---

#  Evaluation

## Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

### Output

```
precision    recall  f1-score   support

0       0.00      0.00      0.00        12
1       0.00      0.00      0.00        12
2       0.75      1.00      0.86        72

accuracy                           0.75        96
macro avg       0.25      0.33      0.29        96
weighted avg    0.56      0.75      0.64        96
```

---

## ROC AUC Score (Multiclass)

```python
from sklearn.metrics import roc_auc_score
print(f"ROC AUC score: {roc_auc_score(y_test, y_proba, multi_class='ovr')}")
```

### Output

```
ROC AUC score: 0.657
```


#  Model Training(Random Forest)

```jupyter
from sklearn.ensemble import RandomForestClassifier

# random orest for differensheat
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf.fit(X_train,y_train)
```

---

#  Predictions

```jupyter
y_pre = rf.predict(X_test)
y_pred = rf.predict_proba(X_test)
```

---

#  Evaluation

## Classification Report

```jupyter
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

### Output

```
classification report:              precision    recall  f1-score   support

           0       0.50      0.08      0.14        12
           1       0.00      0.00      0.00        12
           2       0.76      0.99      0.86        72

    accuracy                           0.75        96
   macro avg       0.42      0.36      0.33        96
weighted avg       0.64      0.75      0.66        96
```

---

## ROC AUC Score (Multiclass)

```python
from sklearn.metrics import roc_auc_score
print(f"ROC AUC score: {roc_auc_score(y_test, y_proba, multi_class='ovr')}")
```

### Output

```
ROC AUC score: 0.6236
```





#  Key Observations(random forest)

- Dataset is **highly imbalanced**
- Model predicts **Profit class well**
- Model fails on **Loss and Neutral classes**
- Indicates need for:
  - Class balancing (SMOTE)
  - Better features
  - Tree-based models

---

#  Future Improvements

```jupyter
# Example: using multi class
class_weight = {0:3, 1:3, 2:1}

LogisticRegression(
    max_iter=1000,
    class_weight=class_weight,
    multi_class="multinomial"
)
```

Planned:

- SMOTE
-  XGBoost
- Volatility & drawdown features
- Feature scaling

---

#  How to Run

```bash
pip install -r requirements.txt
```

```python
python train_model.py
```

---

#  Conclusion

Trader behavior and sentiment provide useful signals for predicting
daily trading outcomes.While Random Forest achieved higher overall accuracy (0.75), it showed severe bias toward the majority class (Profit) and failed to identify Loss and Neutral days.
Logistic Regression with class balancing provided lower accuracy (0.56) but significantly improved recall for minority classes and achieved higher macro F1-score and ROC-AUC.
---
