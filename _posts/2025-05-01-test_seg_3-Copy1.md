---
layout: single
title:  "[dacon] 신용카드 고객 세그먼트 분류 Ai 경진대회 [1,384 中 34]"
categories: coding
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.65rem !important;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


### Import



```python
import pandas as pd
import numpy as np
import gc

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
```

### Data Load



```python
# 데이터 분할(폴더) 구분
data_splits = ["train", "test"]

# 각 데이터 유형별 폴더명, 파일 접미사, 변수 접두어 설정
data_categories = {
    "회원정보": {"folder": "1.회원정보", "suffix": "회원정보", "var_prefix": "customer"},
    "신용정보": {"folder": "2.신용정보", "suffix": "신용정보", "var_prefix": "credit"},
    "승인매출정보": {"folder": "3.승인매출정보", "suffix": "승인매출정보", "var_prefix": "sales"},
    "청구정보": {"folder": "4.청구입금정보", "suffix": "청구정보", "var_prefix": "billing"},
    "잔액정보": {"folder": "5.잔액정보", "suffix": "잔액정보", "var_prefix": "balance"},
    "채널정보": {"folder": "6.채널정보", "suffix": "채널정보", "var_prefix": "channel"},
    "마케팅정보": {"folder": "7.마케팅정보", "suffix": "마케팅정보", "var_prefix": "marketing"},
    "성과정보": {"folder": "8.성과정보", "suffix": "성과정보", "var_prefix": "performance"}
}

# 2018년 7월부터 12월까지의 월 리스트
months = ['07', '08', '09', '10', '11', '12']

for split in data_splits:
    for category, info in data_categories.items():
        folder = info["folder"]
        suffix = info["suffix"]
        var_prefix = info["var_prefix"]
        
        for month in months:
            # 파일명 형식: 2018{month}_{split}_{suffix}.parquet
            file_path = f"C:/Users/pc/OneDrive/바탕 화면/open/{split}/{folder}/2018{month}_{split}_{suffix}.parquet"
            # 변수명 형식: {var_prefix}_{split}_{month}
            variable_name = f"{var_prefix}_{split}_{month}"
            globals()[variable_name] = pd.read_parquet(file_path)
            print(f"{variable_name} is loaded from {file_path}")

gc.collect()
```

<pre>
customer_train_07 is loaded from C:/Users/pc/OneDrive/바탕 화면/open/train/1.회원정보/201807_train_회원정보.parquet
customer_train_08 is loaded from C:/Users/pc/OneDrive/바탕 화면/open/train/1.회원정보/201808_train_회원정보.parquet
....
performance_test_11 is loaded from C:/Users/pc/OneDrive/바탕 화면/open/test/8.성과정보/201811_test_성과정보.parquet
performance_test_12 is loaded from C:/Users/pc/OneDrive/바탕 화면/open/test/8.성과정보/201812_test_성과정보.parquet
</pre>
<pre>
0
</pre>
### Data Preprocessing(1) : Concat & Merge



```python
# 데이터 유형별 설정 
info_categories = ["customer", "credit", "sales", "billing", "balance", "channel", "marketing", "performance"]

# 월 설정
months = ['07', '08', '09', '10', '11', '12']
```


```python
#### Train ####

# 각 유형별로 월별 데이터를 합쳐서 새로운 변수에 저장
train_dfs = {}

import gc
import pandas as pd
import numpy as np

# 결과 저장할 딕셔너리
train_dfs = {}

# 다운캐스트 함수 정의
def optimize_df(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df
 
# 반복 병합
for prefix in info_categories:
    combined_df = pd.DataFrame()  # 임시 병합용 빈 DF
    for month in months:
        var_name = f"{prefix}_train_{month}"
        if var_name in globals():
            df = globals()[var_name]
            df = optimize_df(df)  # 다운캐스팅
            combined_df = pd.concat([combined_df, df], axis=0, ignore_index=True)
            del globals()[var_name]  # 사용한 DF 메모리에서 제거
            gc.collect()
        else:
            print(f"{var_name} 없음")
    train_dfs[f"{prefix}_train_df"] = combined_df
    print(f"{prefix}_train_df 완성: {combined_df.shape}")

customer_train_df = train_dfs["customer_train_df"]
credit_train_df   = train_dfs["credit_train_df"]
sales_train_df    = train_dfs["sales_train_df"]
billing_train_df  = train_dfs["billing_train_df"]
balance_train_df  = train_dfs["balance_train_df"]
channel_train_df  = train_dfs["channel_train_df"]
marketing_train_df= train_dfs["marketing_train_df"]
performance_train_df = train_dfs["performance_train_df"]

gc.collect()
```

<pre>
customer_train_df 완성: (2400000, 78)
credit_train_df 완성: (2400000, 42)
sales_train_df 완성: (2400000, 406)
billing_train_df 완성: (2400000, 46)
balance_train_df 완성: (2400000, 82)
channel_train_df 완성: (2400000, 105)
marketing_train_df 완성: (2400000, 64)
performance_train_df 완성: (2400000, 49)
</pre>
<pre>
0
</pre>

```python
#### Test ####

# test 데이터에 대해 train과 동일한 방법 적용
test_dfs = {}

# 다운캐스트 함수 정의
def optimize_df(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

# 반복 병합
for prefix in info_categories:
    combined_df = pd.DataFrame()  # 임시 병합용 빈 DF
    for month in months:
        var_name = f"{prefix}_test_{month}"
        if var_name in globals():
            df = globals()[var_name]
            df = optimize_df(df)  # 다운캐스팅
            combined_df = pd.concat([combined_df, df], axis=0, ignore_index=True)
            del globals()[var_name]  # 사용한 DF 메모리에서 제거
            gc.collect()
        else:
            print(f"{var_name} 없음")
    test_dfs[f"{prefix}_test_df"] = combined_df
    print(f"{prefix}_test_df 완성: {combined_df.shape}")

customer_test_df = test_dfs["customer_test_df"]
credit_test_df   = test_dfs["credit_test_df"]
sales_test_df    = test_dfs["sales_test_df"]
billing_test_df  = test_dfs["billing_test_df"]
balance_test_df  = test_dfs["balance_test_df"]
channel_test_df  = test_dfs["channel_test_df"]
marketing_test_df= test_dfs["marketing_test_df"]
performance_test_df = test_dfs["performance_test_df"]

gc.collect()
```

<pre>
customer_test_df 완성: (600000, 77)
credit_test_df 완성: (600000, 42)
sales_test_df 완성: (600000, 406)
billing_test_df 완성: (600000, 46)
balance_test_df 완성: (600000, 82)
channel_test_df 완성: (600000, 105)
marketing_test_df 완성: (600000, 64)
performance_test_df 완성: (600000, 49)
</pre>
<pre>
0
</pre>

```python
#### Train ####
# 두 데이터프레임의 기준년월을 int64로, ID를 str로 통일
customer_train_df['기준년월'] = customer_train_df['기준년월'].astype('int64')
credit_train_df['기준년월'] = credit_train_df['기준년월'].astype('int64')

customer_train_df['ID'] = customer_train_df['ID'].astype(str)
credit_train_df['ID'] = credit_train_df['ID'].astype(str)


train_df = customer_train_df.merge(credit_train_df, on=['기준년월', 'ID'], how='left')
print("Step1 저장 완료: train_step1, shape:", train_df.shape)
del customer_train_df, credit_train_df
gc.collect()

# 이후 merge할 데이터프레임 이름과 단계 정보를 리스트에 저장
merge_list = [
    ("sales_train_df",    "Step2"),
    ("billing_train_df",  "Step3"),
    ("balance_train_df",  "Step4"),
    ("channel_train_df",  "Step5"),
    ("marketing_train_df","Step6"),
    ("performance_train_df", "최종")
]

# 나머지 단계 merge
for df_name, step in merge_list:
    # globals()로 동적 변수 접근하여 merge 수행
    train_df = train_df.merge(globals()[df_name], on=['기준년월', 'ID'], how='left')
    print(f"{step} 저장 완료: train_{step}, shape:", train_df.shape)
    # 사용한 변수는 메모리 해제를 위해 삭제
    del globals()[df_name]
    gc.collect()
```

<pre>
Step1 저장 완료: train_step1, shape: (2400000, 118)
Step2 저장 완료: train_Step2, shape: (2400000, 522)
Step3 저장 완료: train_Step3, shape: (2400000, 566)
Step4 저장 완료: train_Step4, shape: (2400000, 646)
Step5 저장 완료: train_Step5, shape: (2400000, 749)
Step6 저장 완료: train_Step6, shape: (2400000, 811)
최종 저장 완료: train_최종, shape: (2400000, 858)
</pre>

```python
#### Test ####
customer_test_df['기준년월'] = customer_test_df['기준년월'].astype('int64')
credit_test_df['기준년월'] = credit_test_df['기준년월'].astype('int64')

customer_test_df['ID'] = customer_test_df['ID'].astype(str)
credit_test_df['ID'] = credit_test_df['ID'].astype(str)


test_df = customer_test_df.merge(credit_test_df, on=['기준년월', 'ID'], how='left')
print("Step1 저장 완료: test_step1, shape:", test_df.shape)
del customer_test_df, credit_test_df
gc.collect()

# 이후 merge할 데이터프레임 이름과 단계 정보를 리스트에 저장
merge_list = [
    ("sales_test_df",    "Step2"),
    ("billing_test_df",  "Step3"),
    ("balance_test_df",  "Step4"),
    ("channel_test_df",  "Step5"),
    ("marketing_test_df","Step6"),
    ("performance_test_df", "최종")
]
 
# 나머지 단계 merge
for df_name, step in merge_list:
    # globals()로 동적 변수 접근하여 merge 수행
    test_df = test_df.merge(globals()[df_name], on=['기준년월', 'ID'], how='left')
    print(f"{step} 저장 완료: test_{step}, shape:", test_df.shape)
    # 사용한 변수는 메모리 해제를 위해 삭제
    del globals()[df_name]
    gc.collect()
```

<pre>
Step1 저장 완료: test_step1, shape: (600000, 117)
Step2 저장 완료: test_Step2, shape: (600000, 521)
Step3 저장 완료: test_Step3, shape: (600000, 565)
Step4 저장 완료: test_Step4, shape: (600000, 645)
Step5 저장 완료: test_Step5, shape: (600000, 748)
Step6 저장 완료: test_Step6, shape: (600000, 810)
최종 저장 완료: test_최종, shape: (600000, 857)
</pre>
### Data 결측치 



```python
# 결측값이 있는 컬럼만 필터링하여 출력

# train
na_val_train = train_df.isna().sum()
na_col_train = na_val_train[na_val_train > 0]

# test
na_val_test = test_df.isna().sum()
na_col_test = na_val_test[na_val_test > 0]


# 결측값이 있는 컬럼 출력
print(f"train \n {na_col_train} \n \n test \n {na_col_test}")

gc.collect()

#결측값 데이터 제거
train_df = train_df.dropna(axis=1)
test_df = test_df.dropna(axis=1)
```

<pre>
train 
 가입통신회사코드           387570
직장시도명              244969
_1순위신용체크구분          27950
_2순위신용체크구분         958115
최종유효년월_신용_이용가능     210447
최종유효년월_신용_이용       534231
최종카드발급일자            41965
RV신청일자            1951236
RV전환가능여부            29473
_1순위업종             539992
_2순위업종             912725
_3순위업종            1107898
_1순위쇼핑업종           922663
_2순위쇼핑업종          1135042
_3순위쇼핑업종          1312267
_1순위교통업종          1164494
_2순위교통업종          1656423
_3순위교통업종          2045455
_1순위여유업종          1987260
_2순위여유업종          2302286
_3순위여유업종          2377725
_1순위납부업종          1216263
_2순위납부업종          2033640
_3순위납부업종          2310187
최종카드론_금융상환방식코드    1958126
최종카드론_신청경로코드      1958226
최종카드론_대출일자        1988330
연체일자_B0M          2394336
OS구분코드            1633566
혜택수혜율_R3M          488746
혜택수혜율_B0M          555522
dtype: int64 
 
 test 
 가입통신회사코드           97083
직장시도명              62233
_1순위신용체크구분          7231
_2순위신용체크구분        239888
최종유효년월_신용_이용가능     53149
최종유효년월_신용_이용      134817
최종카드발급일자           10854
RV신청일자            487518
RV전환가능여부            7634
_1순위업종            136575
_2순위업종            231087
_3순위업종            279806
_1순위쇼핑업종          232771
_2순위쇼핑업종          286358
_3순위쇼핑업종          330955
_1순위교통업종          292960
_2순위교통업종          414730
_3순위교통업종          510729
_1순위여유업종          498313
_2순위여유업종          575821
_3순위여유업종          594565
_1순위납부업종          305994
_2순위납부업종          510271
_3순위납부업종          578711
최종카드론_금융상환방식코드    489771
최종카드론_신청경로코드      489783
최종카드론_대출일자        497538
연체일자_B0M          598605
OS구분코드            407652
혜택수혜율_R3M         123715
혜택수혜율_B0M         140796
dtype: int64
</pre>
### Data Preprocessing(2) : Encoding



```python
feature_cols = [col for col in train_df.columns if col not in ["ID", "Segment","기준년월"]]

X = train_df[feature_cols].copy()
y = train_df["Segment"].copy()

# 타깃 라벨 인코딩
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
```


```python
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

X_test = test_df.copy()

encoders = {}  # 각 컬럼별 encoder 저장

for col in categorical_features:
    le_train = LabelEncoder()
    X[col] = le_train.fit_transform(X[col])
    encoders[col] = le_train
    unseen_labels_val = set(X_test[col]) - set(le_train.classes_)
    if unseen_labels_val:
        le_train.classes_ = np.append(le_train.classes_, list(unseen_labels_val))
    X_test[col] = le_train.transform(X_test[col])
```


```python
gc.collect()
```

<pre>
0
</pre>
### Train



```python
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import early_stopping, log_evaluation

# ========================
# 1. 중요도 0인 Feature 제거
# ========================
print("▶ Feature Selection 시작...")

# 간단한 모델로 전체 데이터에 대해 feature importance 측정
temp_model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    device='cpu',
    random_state=42,
    n_estimators=100
)

temp_model.fit(X, y_encoded)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': temp_model.feature_importances_
})

# 중요도 0인 feature 제거
zero_importance = importance[importance['Importance'] == 0]['Feature'].tolist()
print(f"삭제할 Feature 수: {len(zero_importance)}개")

X_selected = X.drop(columns=zero_importance)  # 수정: 원본 X 건들지 않고 복사본 사용

# ========================
# 2. Stratified KFold 진행
# ========================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

train_f1_scores = []
val_f1_scores = []

feature_importances = pd.DataFrame()
feature_importances['Feature'] = X_selected.columns

for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y_encoded)):
    print(f"\n▶ Fold {fold+1}")

    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # LightGBM 분류모델 사용
    model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        device='cpu',
        random_state=42,
        learning_rate=0.05,
        n_estimators=1000,
        min_data_in_leaf=100,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=6,
        reg_alpha=5.0,
        reg_lambda=5.0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=100)
        ]
    )

    # 훈련 데이터 예측 및 F1
    train_pred = model.predict(X_train)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    train_f1_scores.append(train_f1)

    # 검증 데이터 예측 및 F1
    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average='macro')
    val_f1_scores.append(val_f1)

    print(f"Train F1 Score (Fold {fold+1}): {train_f1:.5f}")
    print(f"Validation F1 Score (Fold {fold+1}): {val_f1:.5f}")

    # feature importance 누적
    fold_importance = pd.DataFrame({
        'Feature': X_selected.columns,
        f'Fold_{fold+1}': model.feature_importances_
    })

    feature_importances = feature_importances.merge(fold_importance, on='Feature', how='left')

# ========================
# 3. 결과 정리
# ========================
feature_cols = [f'Fold_{i+1}' for i in range(n_splits)]
feature_importances['Average'] = feature_importances[feature_cols].mean(axis=1)
feature_importances = feature_importances.sort_values(by='Average', ascending=False)

print("\n=== 최종 결과 ===")
print(f"평균 Train F1 Score: {np.mean(train_f1_scores):.5f}")
print(f"평균 Validation F1 Score: {np.mean(val_f1_scores):.5f}")
print("\n상위 중요 Features:")
print(feature_importances.head(20))
```

<pre>
▶ Feature Selection 시작...
</pre>
<pre>
C:\Users\pc\anaconda3\Lib\site-packages\joblib\externals\loky\backend\context.py:136: UserWarning: Could not find the number of physical cores for the following reason:
[WinError 2] 지정된 파일을 찾을 수 없습니다
Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.
  warnings.warn(
  File "C:\Users\pc\anaconda3\Lib\site-packages\joblib\externals\loky\backend\context.py", line 257, in _count_physical_cores
    cpu_info = subprocess.run(
               ^^^^^^^^^^^^^^^
  File "C:\Users\pc\anaconda3\Lib\subprocess.py", line 548, in run
    with Popen(*popenargs, **kwargs) as process:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\pc\anaconda3\Lib\subprocess.py", line 1026, in __init__
    self._execute_child(args, executable, preexec_fn, close_fds,
  File "C:\Users\pc\anaconda3\Lib\subprocess.py", line 1538, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
</pre>
<pre>
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 10.915487 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 82045
[LightGBM] [Info] Number of data points in the train set: 2400000, number of used features: 707
[LightGBM] [Info] Start training from score -7.811623
[LightGBM] [Info] Start training from score -9.721166
[LightGBM] [Info] Start training from score -2.934402
[LightGBM] [Info] Start training from score -1.927459
[LightGBM] [Info] Start training from score -0.222075
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
삭제할 Feature 수: 251개

▶ Fold 1
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 7.735359 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 81245
[LightGBM] [Info] Number of data points in the train set: 1920000, number of used features: 573
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Info] Start training from score -7.812395
[LightGBM] [Info] Start training from score -9.714246
[LightGBM] [Info] Start training from score -2.934402
[LightGBM] [Info] Start training from score -1.927461
[LightGBM] [Info] Start training from score -0.222075
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
Training until validation scores don't improve for 100 rounds
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[100]	valid_0's multi_logloss: 0.251832
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[200]	valid_0's multi_logloss: 0.227385
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[300]	valid_0's multi_logloss: 0.214453
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[400]	valid_0's multi_logloss: 0.205457
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[500]	valid_0's multi_logloss: 0.198184
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[600]	valid_0's multi_logloss: 0.192197
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[700]	valid_0's multi_logloss: 0.187031
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[800]	valid_0's multi_logloss: 0.182563
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[900]	valid_0's multi_logloss: 0.178772
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[1000]	valid_0's multi_logloss: 0.175107
Did not meet early stopping. Best iteration is:
[1000]	valid_0's multi_logloss: 0.175107
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
Train F1 Score (Fold 1): 0.91176
Validation F1 Score (Fold 1): 0.78549

▶ Fold 2
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 7.372677 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 81071
[LightGBM] [Info] Number of data points in the train set: 1920000, number of used features: 571
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Info] Start training from score -7.811109
[LightGBM] [Info] Start training from score -9.722904
[LightGBM] [Info] Start training from score -2.934402
[LightGBM] [Info] Start training from score -1.927461
[LightGBM] [Info] Start training from score -0.222075
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
Training until validation scores don't improve for 100 rounds
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[1000]	valid_0's multi_logloss: 0.176445
Did not meet early stopping. Best iteration is:
[1000]	valid_0's multi_logloss: 0.176445
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
Train F1 Score (Fold 2): 0.91396
Validation F1 Score (Fold 2): 0.75901

▶ Fold 3
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 7.555204 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 81293
[LightGBM] [Info] Number of data points in the train set: 1920000, number of used features: 572
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Info] Start training from score -7.811109
[LightGBM] [Info] Start training from score -9.722904
[LightGBM] [Info] Start training from score -2.934402
[LightGBM] [Info] Start training from score -1.927457
[LightGBM] [Info] Start training from score -0.222076
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
Training until validation scores don't improve for 100 rounds
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[1000]	valid_0's multi_logloss: 0.175797
Did not meet early stopping. Best iteration is:
[1000]	valid_0's multi_logloss: 0.175797
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
Train F1 Score (Fold 3): 0.91487
Validation F1 Score (Fold 3): 0.78035

▶ Fold 4
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 7.938806 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 81311
[LightGBM] [Info] Number of data points in the train set: 1920000, number of used features: 572
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Info] Start training from score -7.811109
[LightGBM] [Info] Start training from score -9.722904
[LightGBM] [Info] Start training from score -2.934402
[LightGBM] [Info] Start training from score -1.927457
[LightGBM] [Info] Start training from score -0.222076
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
Training until validation scores don't improve for 100 rounds
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[1000]	valid_0's multi_logloss: 0.175941
Did not meet early stopping. Best iteration is:
[1000]	valid_0's multi_logloss: 0.175941
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
Train F1 Score (Fold 4): 0.91114
Validation F1 Score (Fold 4): 0.75663

▶ Fold 5
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 7.940784 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 81320
[LightGBM] [Info] Number of data points in the train set: 1920000, number of used features: 573
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Info] Start training from score -7.812395
[LightGBM] [Info] Start training from score -9.722904
[LightGBM] [Info] Start training from score -2.934402
[LightGBM] [Info] Start training from score -1.927457
[LightGBM] [Info] Start training from score -0.222075
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
Training until validation scores don't improve for 100 rounds
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
....
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[1000]	valid_0's multi_logloss: 0.175843
Did not meet early stopping. Best iteration is:
[1000]	valid_0's multi_logloss: 0.175843
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
Train F1 Score (Fold 5): 0.91298
Validation F1 Score (Fold 5): 0.74504

=== 최종 결과 ===
평균 Train F1 Score: 0.91294
평균 Validation F1 Score: 0.76530

상위 중요 Features:
                Feature  Fold_1  Fold_2  Fold_3  Fold_4  Fold_5  Average
115       이용금액_일시불_R12M    3756    3796    3732    3846    3813   3788.6
55             월상환론한도금액    3385    3207    3220    3160    3394   3273.2
7               입회일자_신용    2708    2789    2791    2715    2896   2779.8
356          정상청구원금_B5M    1740    1775    1814    1788    1777   1778.8
460           평잔_일시불_6M    1686    1687    1687    1739    1634   1686.6
287       이용금액_오프라인_B0M    1436    1531    1542    1549    1506   1512.8
52             카드이용한도금액    1473    1382    1429    1407    1353   1408.8
348          정상청구원금_B0M    1360    1451    1384    1325    1407   1385.4
106        이용건수_신용_R12M    1368    1346    1380    1393    1397   1376.8
121        이용금액_체크_R12M    1317    1363    1307    1319    1331   1327.4
38           _1순위카드이용건수    1270    1313    1337    1366    1313   1319.8
123     최대이용금액_일시불_R12M    1186    1258    1260    1258    1265   1245.4
79            최종이용일자_CA    1032    1008     988     986     992   1001.2
364  연속유실적개월수_기본_24M_카드     984    1006     971    1018     985    992.8
83            최종이용일자_할부     990     961     969     988     951    971.8
146        이용금액_일시불_R6M    1015     927     985     946     977    970.0
8            입회경과개월수_신용    1021     884     873     952     887    923.4
129      최대이용금액_체크_R12M     889     899     921     901     926    907.2
564           변동률_일시불평잔     865     953     909     911     893    906.2
283       이용금액_오프라인_R3M     880     920     914     884     876    894.8
</pre>
### Predict



```python
X_test.drop(columns=['ID','기준년월'],inplace=True)
```


```python
# 중요도 0인 피처 제거 
X_test = X_test.drop(columns=zero_importance)

# GPU 예측시 DMatrix로 바꿔 예측
y_test_pred = model.predict(X_test)
y_test_pred_labels = le_target.inverse_transform(y_test_pred)

test_data = test_df.copy()
test_data["pred_label"] = y_test_pred_labels
```

<pre>
[LightGBM] [Warning] min_data_in_leaf is set=100, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=100
</pre>
### Submission



```python
submission = tㄴest_data.groupby("ID")["pred_label"] \
    .agg(lambda x: x.value_counts().idxmax()) \
    .reset_index() 

submission.columns = ["ID", "Segment"]
```


```python
submission.to_csv('./segment_3.csv',index=False)
```
