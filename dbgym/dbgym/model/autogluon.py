from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from dbgym.config import Config


class AutogluonModel:
    def __init__(self, save_path: Path, predictor_target):
        self.save_path: Path = save_path
        self.predictor_target = predictor_target
        self.unit = self.predictor_target.split(" ")[-1]
        self.predictor: TabularPredictor = TabularPredictor(label=self.predictor_target, path=str(self.save_path))

    def try_load(self) -> bool:
        try:
            self.predictor.load(str(self.save_path))
            # TODO(WAN): leak of autogluon impl details, but sometimes fit doesn't save?
            return self.predictor._learner.is_fit
        except FileNotFoundError:
            return False

    def train(self, dataset: TabularDataset, time_limit=Config.AUTOGLUON_TIME_LIMIT_S):
        print("Training:", dataset.columns)
        self.predictor.fit(dataset, time_limit=time_limit)
        pd.Series({"Training Time (s)": time_limit}).to_pickle(self.save_path / "training_time.pkl")
        self.predictor.save()

    def eval(self, dataset: TabularDataset) -> pd.DataFrame:
        eval_dataset = dataset.drop(columns=[self.predictor_target])
        print("Eval:", eval_dataset.columns)
        y_pred = self.predictor.predict(eval_dataset)
        eval_df = pd.concat([y_pred, dataset[self.predictor_target]], axis=1)
        eval_df.columns = [f"Predicted {self.unit}", f"Actual {self.unit}"]
        eval_df[f"Diff {self.unit}"] = (eval_df[f"Predicted {self.unit}"] - eval_df[f"Actual {self.unit}"]).abs()
        eval_df["q_err"] = np.nan_to_num(
            np.maximum(
                eval_df[f"Predicted {self.unit}"] / eval_df[f"Actual {self.unit}"],
                eval_df[f"Actual {self.unit}"] / eval_df[f"Predicted {self.unit}"],
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
