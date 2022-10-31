import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor

experiments = [
    (("tgtd", "OnlyTrain"), ("tdtd", "OnlyTest")),
    (("tgtd", "OnlyTrain"), ("tgtd", "OnlyTest")),
    (("tgtd", "PerfectPrediction"), ("tdtd", "OnlyTest")),
    (("tgtd", "PerfectPrediction"), ("tgtd", "OnlyTest")),
    (("tdtd", "OnlyTrain"), ("tdtd", "OnlyTest")),
    (("tdtd", "PerfectPrediction"), ("tdtd", "OnlyTest")),
]

time_limit = 300
retrain = True

def flatten(df):
    flatteneds = []
    for col in ["Children Observation Indexes", "Features", "Query Hash"]:
        col_df = pd.DataFrame(df[col].tolist(), index=df.index)
        col_df = col_df.rename(columns=lambda num: f"{col}_{num}")
        flatteneds.append(col_df)
    for col in ["Node Type", "Observation Index", "Query Num", "Actual Total Time (us)"]:
        flatteneds.append(df[col])
    return pd.concat(flatteneds, axis=1)

for experiment in experiments:
    (train_name, train_variation), (test_name, test_variation) = experiment
    save_path = f"./autogluon/{train_name}-{train_variation}_{test_name}-{test_variation}/"
    train_path = f"/home/wanshenl/git/dbgym/artifact/experiment/{train_name}/{train_variation}/observations_0.parquet"
    test_path = f"/home/wanshenl/git/dbgym/artifact/experiment/{test_name}/{test_variation}/observations_0.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    train_data = TabularDataset(flatten(train_df))
    test_data = TabularDataset(flatten(test_df))

    if retrain:
        predictor = TabularPredictor(label="Actual Total Time (us)", path=save_path).fit(train_data, time_limit=time_limit)
    else:
        predictor = TabularPredictor.load(save_path)
    
    # Predict.
    y_pred = predictor.predict(test_data)
    eval_df = pd.concat([y_pred, test_data["Actual Total Time (us)"]], axis=1)
    eval_df.columns = ["Predicted Latency (us)", "Actual Latency (us)"]
    
    metrics_df = eval_df.copy()
    metrics_df["diff (us)"] = (metrics_df["Predicted Latency (us)"] - metrics_df["Actual Latency (us)"]).abs()
    metrics_df["q_err"] = np.nan_to_num(
        np.maximum(
            metrics_df["Predicted Latency (us)"] / metrics_df["Actual Latency (us)"],
            metrics_df["Actual Latency (us)"] / metrics_df["Predicted Latency (us)"]
        ),
        nan=np.inf
    )
    print(metrics_df["diff (us)"].describe())
    print(metrics_df["q_err"].describe())
