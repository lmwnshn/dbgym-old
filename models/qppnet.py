"""
TODO(WAN): docs

Based off rabbit721's QPPNet implementation
https://github.com/rabbit721/QPPNet
"""

from pathlib import Path
from typing import TypeAlias

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
        batch_size: int = _PAPER_BATCH_SIZE,
        num_epochs: int = _PAPER_NUM_EPOCHS,
    ):
        self._train_df: pd.DataFrame = train_df
        self._test_df: pd.DataFrame = test_df
        self._save_folder = save_folder

        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._random_state = np.random.RandomState(15721)

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

    def train(self, num_epochs=None, force_stratify=True, epoch_save_interval=None):
        df = self._train_df

        num_epochs = self._num_epochs if num_epochs is None else num_epochs
        for epoch in tqdm(
            range(1, num_epochs + 1),
            desc=f"Training QPPNet for {num_epochs} epochs.",
        ):
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
                sizes = df.groupby("Query Hash").size()
                must_take = df[df["Query Hash"].isin(sizes[sizes == 1].keys())].index
                remaining = df[~df.index.isin(must_take)]
                train, _ = train_test_split(
                    remaining,
                    train_size=self._batch_size,
                    random_state=self._random_state,
                    stratify=remaining["Query Hash"],
                )
                query_nums = pd.concat([train, df.iloc[must_take]])[
                    "Query Num"
                ].unique()
                batch = df[df["Query Num"].isin(query_nums)]
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
                    gdf["Actual Total Time"],
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
                batch_loss = torch.sqrt(
                    torch.mean(loss_sum / num_losses_total)
                )  # Eq. (7)
                scaling_factor = num_losses_node_type / num_losses_total
                batch_loss.backward(torch.ones_like(batch_loss) * scaling_factor)
                self._optimizers[neural_unit_id].step()
                self._schedulers[neural_unit_id].step()
                pbar.update()
            pbar.close()

            if (epoch_save_interval is not None) and (epoch % epoch_save_interval == 0):
                evaluate_df = self.evaluate(leave_tqdm=False)
                rmse, rel_err, mae, rq = self.compute_metrics(evaluate_df)
                suffix = (
                    f"rmse_{rmse:.4f}_relerr_{rel_err:.4f}_mae_{mae:.4f}rq_{rq:.4f}]"
                )
                self.save_weights(epoch, suffix)

    def evaluate(self, leave_tqdm=True):
        with torch.no_grad():
            df = self._test_df
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
                for node_type, obs_idx, children_idxs, features, actual_time in zip(
                    gdf["Node Type"],
                    gdf["Observation Index"],
                    gdf["Children Observation Indexes"],
                    gdf["Features"],
                    gdf["Actual Total Time"],
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
                "Estimated Latency": estimated_latencies,
                "Actual Latency": actual_latencies,
            }
        )

    def save_weights(self, epoch: int, model_suffix: str):
        for neural_unit_id, neural_unit in self._neural_units.items():
            node_type, num_children = neural_unit_id
            filename = f"id_{node_type}-{num_children}_epoch_{epoch}_{model_suffix}.pth"
            path = self._save_folder / filename
            torch.save(neural_unit.state_dict(), path)

    def load_weights(self, epoch: int):
        for neural_unit_id in self._neural_units.keys():
            node_type, num_children = neural_unit_id
            filename = f"id_{node_type}-{num_children}_epoch_{epoch}*.pth"
            matching_files = list(self._save_folder.glob(filename))
            assert len(matching_files) == 1
            state_dict = torch.load(matching_files[0])
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
            (evaluate_df["Actual Latency"] - evaluate_df["Estimated Latency"]) ** 2
        ).mean() ** 0.5

    @staticmethod
    def calculate_relative_error(evaluate_df):
        return (
            1
            / len(evaluate_df)
            * (
                (
                    (
                        evaluate_df["Actual Latency"] - evaluate_df["Estimated Latency"]
                    ).abs()
                )
                / evaluate_df["Actual Latency"]
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
                        evaluate_df["Actual Latency"] - evaluate_df["Estimated Latency"]
                    ).abs()
                )
            ).sum()
        )

    @staticmethod
    def calculate_rq(evaluate_df):
        # From the paper,
        # > R(q) = maximum(ratio between the actual and the predicted, ratio between the predicted and the actual)
        # > Intuitively, the R(q) value represents the "factor" by which a particular estimate was off.
        a = evaluate_df["Estimated Latency"] / evaluate_df["Actual Latency"]
        b = evaluate_df["Actual Latency"] / evaluate_df["Estimated Latency"]
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
