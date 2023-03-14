from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from dbgym.config import Config


class AutogluonModel:
    def __init__(self, save_path: Path):
        self.save_path: Path = save_path
        self.predictor: TabularPredictor = TabularPredictor(label="Actual Total Time (us)", path=str(self.save_path))

    def try_load(self) -> bool:
        try:
            self.predictor.load(str(self.save_path))
            # TODO(WAN): leak of autogluon impl details, but sometimes fit doesn't save?
            return self.predictor._learner.is_fit
        except FileNotFoundError:
            return False

    def train(self, dataset: TabularDataset, time_limit=Config.AUTOGLUON_TIME_LIMIT_S):
        self.predictor.fit(dataset, time_limit=time_limit)
        pd.Series({"Training Time (s)": time_limit}).to_pickle(self.save_path / "training_time.pkl")
        self.predictor.save()

    def eval(self, dataset: TabularDataset) -> pd.DataFrame:
        y_pred = self.predictor.predict(dataset)
        eval_df = pd.concat([y_pred, dataset["Actual Total Time (us)"]], axis=1)
        eval_df.columns = ["Predicted Latency (us)", "Actual Latency (us)"]
        eval_df["diff (us)"] = (eval_df["Predicted Latency (us)"] - eval_df["Actual Latency (us)"]).abs()
        eval_df["q_err"] = np.nan_to_num(
            np.maximum(
                eval_df["Predicted Latency (us)"] / eval_df["Actual Latency (us)"],
                eval_df["Actual Latency (us)"] / eval_df["Predicted Latency (us)"],
            ),
            nan=np.inf,
        )
        return eval_df

    @staticmethod
    def make_padded_datasets(dfs: list[pd.DataFrame]) -> list[TabularDataset]:
        result = []
        max_lens = {}
        for df in dfs:
            for col in ["Children Observation Indexes", "Features", "Query Hash"]:
                max_lens[col] = max([len(x) for x in df[col].tolist()])
        for df in dfs:
            flatteneds = []
            for col in ["Children Observation Indexes", "Features", "Query Hash"]:
                col_df = pd.DataFrame(df[col].tolist(), index=df.index)
                col_df = col_df.rename(columns=lambda num: f"{col}_{num}")
                for i in range(len(col_df.columns), max_lens[col] + 1):
                    col_df[f"{col}_{i}"] = pd.NA
                flatteneds.append(col_df)
            for col in ["Node Type", "Observation Index", "Query Num", "Actual Total Time (us)"]:
                flatteneds.append(df[col])
            result.append(TabularDataset(pd.concat(flatteneds, axis=1)))
        return result
