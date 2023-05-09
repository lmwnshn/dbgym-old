from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from dbgym.config import Config


class AutogluonModel:
    def __init__(self, save_path: Path, predictor_target):
        self.save_path: Path = save_path
        self.predictor_target = predictor_target
        self.predictor: TabularPredictor = TabularPredictor(label=self.predictor_target, path=str(self.save_path))

    def try_load(self) -> bool:
        try:
            self.predictor.load(str(self.save_path))
            # TODO(WAN): leak of autogluon impl details, but sometimes fit doesn't save?
            return self.predictor._learner.is_fit
        except FileNotFoundError:
            return False

    def _featurize(self, dataset):
        dataset = dataset[["Node Type", "Features", self.predictor_target]]
        return dataset

    def train(self, dataset: TabularDataset, time_limit=Config.AUTOGLUON_TIME_LIMIT_S):
        self.predictor.fit(self._featurize(dataset), time_limit=time_limit)
        pd.Series({"Training Time (s)": time_limit}).to_pickle(self.save_path / "training_time.pkl")
        self.predictor.save()

    def eval(self, dataset: TabularDataset) -> pd.DataFrame:
        y_pred = self.predictor.predict(self._featurize(dataset))
        eval_df = pd.concat([y_pred, dataset[self.predictor_target]], axis=1)
        eval_df.columns = ["Predicted", "Actual"]
        eval_df["Diff"] = (eval_df["Predicted"] - eval_df["Actual"]).abs()
        eval_df["q_err"] = np.nan_to_num(
            np.maximum(
                eval_df["Predicted"] / eval_df["Actual"],
                eval_df["Actual"] / eval_df["Predicted"],
            ),
            nan=np.inf,
        )
        return eval_df

    @staticmethod
    def make_padded_datasets(dfs: list[pd.DataFrame], predictor_target: str) -> list[TabularDataset]:
        assert predictor_target in dfs[0].columns, f"Couldn't find {predictor_target} in {dfs[0].columns}"
        result = []
        max_lens = {}
        for df in dfs:
            for col in ["Features"]:
                max_lens[col] = max([len(x) for x in df[col].tolist()])
        for df in dfs:
            flatteneds = []
            for col in ["Node Type", predictor_target]:
                flatteneds.append(df[col])
            for col in ["Features"]:
                col_df = pd.DataFrame(df[col].tolist(), index=df.index)
                col_df = col_df.rename(columns=lambda num: f"{col}_{num}")
                for i in range(len(col_df.columns), max_lens[col] + 1):
                    col_df[f"{col}_{i}"] = pd.NA
                flatteneds.append(col_df)
            result.append(TabularDataset(pd.concat(flatteneds, axis=1)))
        return result
