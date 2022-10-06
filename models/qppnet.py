"""
TODO(WAN): docs

Based off rabbit721's QPPNet implementation
https://github.com/rabbit721/QPPNet
"""

from pathlib import Path
from typing import Optional, TypeAlias

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

NeuralUnitId: TypeAlias = tuple[str, int]  # Node Type, Num Children

# From the paper.
# > Neural networks
# > Unless otherwise stated, each neural unit had 5 hidden layers, each with 128 neurons each.
# > The data output size was set to d = 32.
# > Rectified linear units (ReLUs [12]) were used as activation functions.
# > Standard stochastic gradient descent (SGD) was used to train the network,
# > with a learning rate of 0.001 and a momentum of 0.9.
# > Training was conducted over 1000 epochs (full passes over the training queries),
# > which consistently produced the reported results.
# > We used the PyTorch [39] library to implement the neural
# > network, and we used its built-in SGD implementation.
_PAPER_NUM_HIDDEN_LAYERS = 5
_PAPER_NUM_NEURONS = 128
_PAPER_DATA_OUTPUT_SIZE = 32
_PAPER_LEARNING_RATE = 1e-3
_PAPER_MOMENTUM = 0.9
_PAPER_NUM_EPOCHS = 1000
_PAPER_BATCH_SIZE = None


class QPPNet:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        save_folder: Path,
        batch_size: Optional[int] = _PAPER_BATCH_SIZE,
        validation_size: Optional[int] = None,
        patience_improvement: float = 1e-3,
        patience: int = 10,
        num_epochs: int = _PAPER_NUM_EPOCHS,
    ):
        self._train_df: pd.DataFrame = train_df
        self._test_df: pd.DataFrame = test_df
        self._save_folder = save_folder

        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._random_state = np.random.RandomState(15721)

        self._validation_size = validation_size
        self._min_validation_loss: np.float32 = np.inf
        self._patience_improvement = patience_improvement
        self._patience = patience

        # Compute dimensions.
        self._dim_dict = self._compute_dim_dict(
            pd.concat([self._train_df, self._test_df])
        )
        self._neural_units: dict[NeuralUnitId, NeuralUnit] = {}
        self._optimizers: dict[NeuralUnitId, Optimizer] = {}
        self._schedulers: dict[NeuralUnitId, StepLR] = {}
        # Initialize neural units.
        for neural_unit_id, input_length in self._dim_dict.items():
            neural_unit = NeuralUnit(neural_unit_id, input_length)
            optimizer = torch.optim.SGD(
                neural_unit.parameters(),
                lr=_PAPER_LEARNING_RATE,
                momentum=_PAPER_MOMENTUM,
            )
            # TODO(WAN): this is an addition from Katrina's reimplementation, which I have kept.
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1000, gamma=0.95
            )
            self._neural_units[neural_unit_id] = neural_unit
            self._optimizers[neural_unit_id] = optimizer
            self._schedulers[neural_unit_id] = scheduler

    @staticmethod
    def split(
        df,
        train_size=None,
        test_size=None,
        random_state: np.random.RandomState | int = 15721,
    ):
        n_all, n_train, n_test = len(df), None, None
        if train_size is not None:
            n_train = train_size if isinstance(train_size, int) else train_size * n_all
        if test_size is not None:
            n_test = test_size if isinstance(test_size, int) else test_size * n_all
        if n_train is None:
            n_train = n_all - test_size
        if n_test is None:
            n_test = n_all - train_size
        assert (
            n_train is not None and n_test is not None
        ), "Need to specify either train size or test size."

        hash_groups = df.groupby("Query Hash")
        sizes = hash_groups.size()
        train_sample = ((sizes / sizes.sum()) * n_train).apply(lambda x: int(max(x, 1)))
        test_sample = ((sizes / sizes.sum()) * n_test).apply(lambda x: int(max(x, 1)))

        train_idxs, test_idxs = [], []
        for group, gdf in hash_groups:
            train_idxs.extend(gdf.sample(n=train_sample[group]).index.values.tolist())
            test_idxs.extend(gdf.sample(n=test_sample[group]).index.values.tolist())

        train_queries = df.iloc[train_idxs]["Query Num"]
        test_queries = df.iloc[test_idxs]["Query Num"]
        train_df = df[df["Query Num"].isin(train_queries)]
        test_df = df[df["Query Num"].isin(test_queries)]
        # assert set(df["Query Hash"].unique()) == set(train_df["Query Hash"].unique())
        # assert set(df["Query Hash"].unique()) == set(test_df["Query Hash"].unique())
        return train_df, test_df

    def train(
        self,
        validation_df=None,
        num_epochs=None,
        force_stratify=True,
        epoch_save_interval=None,
    ):
        df = self._train_df
        if validation_df is None:
            if self._validation_size is not None:
                df, validation_df = self.split(
                    df, test_size=self._validation_size, random_state=self._random_state
                )

        early_stop = False
        patience = self._patience
        num_epochs = self._num_epochs if num_epochs is None else num_epochs
        for epoch in tqdm(
            range(1, num_epochs + 1),
            desc=f"Training QPPNet for {num_epochs} epochs.",
        ):
            if early_stop:
                print(
                    f"Early stopping: epoch {epoch}, min val loss {self._min_validation_loss}."
                )
                break
            # From the paper:
            # > Plan-based batch training
            # > To address these challenges, we propose constructing large batches of randomly sampled query plans.
            # > Within each large batch B, we group together sets of query plans with identical structure,
            # > i.e. we partition B into equivalence classes c_1, c_2, ..., c_n based on plan's tree structure,
            # > such that U_{i=1}^{n} c_{i} = B.
            # With that said, the paper doesn't seem to actually use batch training most of the time.
            # Instead, it takes full passes over all the training queries.
            if self._batch_size is None:
                batch = df
            elif force_stratify:
                batch, _ = self.split(
                    df, train_size=self._batch_size, random_state=self._random_state
                )
            else:
                randomly_sampled_query_plans = np.random.choice(
                    df["Query Num"].unique(), size=self._batch_size, replace=False
                )
                batch = df[df["Query Num"].isin(randomly_sampled_query_plans)]
            equivalence_classes = batch.groupby("Query Hash")

            # Zero out previous gradients.
            for optimizer in self._optimizers.values():
                optimizer.zero_grad()

            # For each class, take a forward pass using this batch.
            pbar = tqdm(
                total=len(batch),
                leave=False,
                desc="Forward pass through batch.",
            )
            class_losses: dict[NeuralUnitId, list] = {}
            for _, gdf in equivalence_classes:
                # Cache the output of neural units as the output of children will be used as input to parent nodes.
                output_cache = {}
                # Sort by observation index descending to generate children first.
                gdf = gdf.sort_values("Observation Index", ascending=False)
                # Forward pass on every model.
                for node_type, obs_idx, children_idxs, features, actual_time in zip(
                    gdf["Node Type"],
                    gdf["Observation Index"],
                    gdf["Children Observation Indexes"],
                    gdf["Features"],
                    gdf["Actual Total Time (us)"],
                ):
                    neural_unit_id = (node_type, len(children_idxs))

                    # Form the input vector for this neural unit.
                    input_vector = []
                    input_vector.extend(features)
                    for child_idx in children_idxs:
                        assert (
                            child_idx in output_cache
                        ), f"While computing {obs_idx}, {child_idx} not cached?"
                        input_vector.extend(output_cache[child_idx])
                    input_vector = torch.tensor(input_vector)

                    # Run the model.
                    model = self._neural_units[neural_unit_id]
                    output = model.forward(input_vector)
                    output_cache[obs_idx] = output

                    # From the paper,
                    # >  The first element of the output vector represents the neural unit's estimation
                    # > of the operator's latency, denoted as p_a[l].
                    # > The remaining d elements represent the data vector, denoted as p_a[d].
                    estimated_latency = output[0]
                    loss = (estimated_latency - actual_time) ** 2

                    class_losses[neural_unit_id] = class_losses.get(neural_unit_id, [])
                    class_losses[neural_unit_id].append(loss)
                    pbar.update()
            pbar.close()

            # Update the gradients.
            pbar = tqdm(
                total=len(class_losses),
                leave=False,
                desc="Backward pass through batch.",
            )
            num_losses_total = sum(len(losses) for losses in class_losses.values())
            for neural_unit_id, losses in class_losses.items():
                # Section 5.1.1.
                num_losses_node_type = len(losses)
                loss_sum = torch.sum(torch.stack(losses))
                # Eq. (7) defines the batch loss (rmse).
                batch_loss = torch.sqrt(torch.mean(loss_sum))
                scaling_factor = num_losses_node_type / num_losses_total
                batch_loss.backward(torch.ones_like(batch_loss) * scaling_factor)
                self._optimizers[neural_unit_id].step()
                self._schedulers[neural_unit_id].step()
                pbar.update()
            pbar.close()

            validation_loss = None
            should_save = (epoch_save_interval is not None) and (
                epoch % epoch_save_interval == 0
            )

            if validation_df is not None:
                evaluate_validation_df = self.evaluate(
                    df=validation_df, leave_tqdm=False
                )
                validation_rmse, _, _, _ = self.compute_metrics(evaluate_validation_df)
                validation_loss = validation_rmse

                if validation_loss > (
                    self._min_validation_loss + self._patience_improvement
                ):
                    # Validation loss is not improving fast enough.
                    patience -= 1
                    if patience == 0:
                        early_stop = True
                else:
                    # Validation loss is improving fast enough, reset patience.
                    patience = self._patience

                if validation_loss < self._min_validation_loss:
                    self._min_validation_loss = validation_loss
                    should_save = True

            if should_save:
                metrics_str = ""
                if validation_loss is not None:
                    metrics_str += f"validationloss_{validation_loss:.3f}"
                self.save_weights(epoch, metrics_str)

    def evaluate(self, df=None, leave_tqdm=True):
        if df is None:
            df = self._test_df
        with torch.no_grad():
            equivalence_classes = df.groupby("Query Hash")

            pbar = tqdm(
                total=len(df),
                leave=leave_tqdm,
                desc="Computing evaluation metrics.",
            )

            observation_indexes = []
            estimated_latencies = []
            actual_latencies = []
            for _, gdf in equivalence_classes:
                # Cache the output of neural units as the output of children will be used as input to parent nodes.
                output_cache = {}
                # Sort by observation index descending to generate children first.
                gdf = gdf.sort_values("Observation Index", ascending=False)
                # Forward pass on every model.
                for (
                    node_type,
                    obs_idx,
                    children_idxs,
                    query_num,
                    features,
                    actual_time,
                ) in zip(
                    gdf["Node Type"],
                    gdf["Observation Index"],
                    gdf["Children Observation Indexes"],
                    gdf["Query Num"],
                    gdf["Features"],
                    gdf["Actual Total Time (us)"],
                ):
                    neural_unit_id = (node_type, len(children_idxs))

                    # Form the input vector for this neural unit.
                    input_vector = []
                    input_vector.extend(features)
                    for child_idx in children_idxs:
                        assert (
                            child_idx in output_cache
                        ), f"While computing {obs_idx}, {child_idx} not cached?"
                        input_vector.extend(output_cache[child_idx])
                    input_vector = torch.tensor(input_vector)

                    # Run the model.
                    model = self._neural_units[neural_unit_id]
                    output = model.forward(input_vector)
                    output_cache[obs_idx] = output

                    # From the paper,
                    # >  The first element of the output vector represents the neural unit's estimation
                    # > of the operator's latency, denoted as p_a[l].
                    # > The remaining d elements represent the data vector, denoted as p_a[d].
                    estimated_latency = output[0].item()

                    observation_indexes.append(obs_idx)
                    estimated_latencies.append(estimated_latency)
                    actual_latencies.append(actual_time)
                    pbar.update()

        pbar.close()
        return pd.DataFrame(
            {
                "Observation Index": observation_indexes,
                "Estimated Latency (us)": estimated_latencies,
                "Actual Latency (us)": actual_latencies,
            }
        )

    def save_weights(self, epoch: int, metrics_str: str):
        for neural_unit_id, neural_unit in self._neural_units.items():
            node_type, num_children = neural_unit_id
            filename = (
                f"id_{node_type}-{num_children}_epoch_{epoch}_m_{metrics_str}_end.pth"
            )
            path = self._save_folder / filename
            torch.save(neural_unit.state_dict(), path)

    def load_weights(self, epoch: int):
        for neural_unit_id in self._neural_units.keys():
            node_type, num_children = neural_unit_id
            filename = f"id_{node_type}-{num_children}_epoch_{epoch}_m_*_end.pth"
            candidates = list(self._save_folder.glob(filename))
            assert len(candidates) == 1
            path = candidates[0]
            state_dict = torch.load(path)
            self._neural_units[neural_unit_id].load_state_dict(state_dict)

    @staticmethod
    def compute_metrics(evaluate_df):
        rmse = QPPNet.calculate_root_mean_square_error(evaluate_df)
        rel_err = QPPNet.calculate_relative_error(evaluate_df)
        mae = QPPNet.calculate_mean_absolute_error(evaluate_df)
        rq = QPPNet.calculate_rq(evaluate_df)
        return rmse, rel_err, mae, rq

    @staticmethod
    def calculate_root_mean_square_error(evaluate_df):
        return (
            (evaluate_df["Actual Latency (us)"] - evaluate_df["Estimated Latency (us)"])
            ** 2
        ).mean() ** 0.5

    @staticmethod
    def calculate_relative_error(evaluate_df):
        return (
            1
            / len(evaluate_df)
            * (
                (
                    (
                        evaluate_df["Actual Latency (us)"]
                        - evaluate_df["Estimated Latency (us)"]
                    ).abs()
                )
                / evaluate_df["Actual Latency (us)"]
            ).sum()
        )

    @staticmethod
    def calculate_mean_absolute_error(evaluate_df):
        return (
            1
            / len(evaluate_df)
            * (
                (
                    (
                        evaluate_df["Actual Latency (us)"]
                        - evaluate_df["Estimated Latency (us)"]
                    ).abs()
                )
            ).sum()
        )

    @staticmethod
    def calculate_rq(evaluate_df):
        # From the paper,
        # > R(q) = maximum(ratio between the actual and the predicted, ratio between the predicted and the actual)
        # > Intuitively, the R(q) value represents the "factor" by which a particular estimate was off.
        a = evaluate_df["Estimated Latency (us)"] / evaluate_df["Actual Latency (us)"]
        b = evaluate_df["Actual Latency (us)"] / evaluate_df["Estimated Latency (us)"]
        return pd.concat([a, b], axis=1).max(axis=1).max()

    @staticmethod
    def _compute_dim_dict(df: pd.DataFrame) -> dict[NeuralUnitId, int]:
        dim_dict = {}
        for node_type, num_children, feature_len in zip(
            df["Node Type"],
            df["Children Observation Indexes"].apply(len),
            df["Features"].apply(len),
        ):
            input_length = feature_len + NeuralUnit.DATA_OUTPUT_SIZE * num_children
            neural_unit_id = (node_type, num_children)
            if neural_unit_id in dim_dict:
                assert (
                    dim_dict[neural_unit_id] == input_length
                ), f"Disagreement on input length?"
            else:
                dim_dict[neural_unit_id] = input_length
        return dim_dict


class NeuralUnit(nn.Module):
    # From the paper: unless otherwise stated,
    # each neural unit had 5 hidden layers, each with 128 neurons each.
    # The data output size was set to d = 32.
    # Rectified linear units were used as activation functions.
    NUM_HIDDEN_LAYERS = _PAPER_NUM_HIDDEN_LAYERS
    NUM_NEURONS = _PAPER_NUM_NEURONS
    DATA_OUTPUT_SIZE = _PAPER_DATA_OUTPUT_SIZE

    def __init__(
        self,
        node_type,
        input_dim,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        hidden_size=NUM_NEURONS,
        output_size=DATA_OUTPUT_SIZE,
    ):
        super().__init__()
        self.node_type = node_type

        model = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers):
            model.append(nn.Linear(hidden_size, hidden_size))
            model.append(nn.ReLU())
        model.append(nn.Linear(hidden_size, output_size))
        model.append(nn.ReLU())
        for layer in model:
            if "weight" in dir(layer):
                nn.init.xavier_uniform_(layer.weight)
        model = nn.Sequential(*model)
        self.model = model

    def forward(self, x):
        return self.model(x)
