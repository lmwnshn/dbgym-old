import os

import pandas as pd
from autogluon.tabular import TabularDataset
from dbgym.config import Config
from dbgym.model.autogluon import AutogluonModel


class GymConfigModel:
    @staticmethod
    def save_model_eval(
        _expt_name: str, _train_data: TabularDataset, _test_data: TabularDataset, predictor_target: str
    ):
        eval_exists = (Config.SAVE_PATH_EVAL / _expt_name).exists()
        eval_newer = eval_exists and os.path.getmtime(Config.SAVE_PATH_EVAL / _expt_name) > os.path.getmtime(
            Config.SAVE_PATH_OBSERVATION / _expt_name / "0.parquet"
        )

        if eval_exists and eval_newer:
            return

        expt_autogluon = AutogluonModel(Config.SAVE_PATH_MODEL / _expt_name, predictor_target=predictor_target)
        if not expt_autogluon.try_load():
            expt_autogluon.train(_train_data)

        (Config.SAVE_PATH_EVAL / _expt_name).mkdir(parents=True, exist_ok=True)
        accuracy = expt_autogluon.eval(_test_data)
        accuracy.to_parquet(Config.SAVE_PATH_EVAL / _expt_name / "accuracy.parquet")

    @staticmethod
    def generate_model(test_df_name: str, train_df_names: list[str]):
        data_dfs = [pd.read_parquet(Config.SAVE_PATH_OBSERVATION / test_df_name / "0.parquet")]
        for name in train_df_names:
            data_dfs.append(pd.read_parquet(Config.SAVE_PATH_OBSERVATION / name / "0.parquet"))

        predictor_target = "Actual Total Time (ms)"
        autogluon_dfs = AutogluonModel.make_padded_datasets(data_dfs, predictor_target=predictor_target)
        test_data = autogluon_dfs[0]
        for name, train_data in zip(train_df_names, autogluon_dfs[1:]):
            GymConfigModel.save_model_eval(name, train_data, test_data, predictor_target=predictor_target)
