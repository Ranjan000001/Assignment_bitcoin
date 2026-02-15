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
# filling mising values
bitcoin['value'] = bitcoin['value'].fillna(bitcoin['value'].median())
bitcoin['classification'] = bitcoin['classification'].fillna(bitcoin['classification'].mode()[0])

```
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
```jupyter
daily_bitcoin.groupby('classification')[['daily_pnl','win_rate']].mean()
daily_bitcoin.groupby('classification')[['trades_per_day', 'avg_trade_size', 'long_short_ratio']].mean()


```

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

## Encoding Sentiment

```jupyter

oe = OrdinalEncoder()
X['classification'] = oe.fit_transform(X[['classification']])
```

---

## Train Test Split

```jupyter
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Handling Missing Values

```jupyter

print(y_train.isnull().sum())
print(y_test.isnull().sum())

y_train = y_train.fillna(y_train.mean())
```

---

#  Model Training

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
ROC AUC score: 0.678
```

---

#  Key Observations

- Dataset is **highly imbalanced**
- Model predicts **Profit class well**
- Model fails on **Loss and Neutral classes**
- Indicates need for:
  - Class balancing (SMOTE)
  - Better features
  - Tree-based models

---

#  Future Improvements

```python
# Example: using class_weight to handle imbalance
lr = LogisticRegression(
    random_state=42,
    max_iter=1000,
    multi_class='multinomial',
    class_weight='balanced'
)
```

Planned:

- SMOTE
- Random Forest / XGBoost
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
daily trading outcomes. Current results show strong class imbalance,
and future work will focus on improving minority class detection.

---
