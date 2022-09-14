from typing import Optional, Union, List, Tuple, Iterable

import gym
import numpy as np
import psutil
from gym.core import RenderFrame, ActType, ObsType
from gym import spaces
from gym.spaces import Space

import random
from dbgym.envs.gym_spec import GymSpec
from dbgym.envs.trainer import PostgresTrainer

from sqlalchemy import create_engine

import threading

from plumbum import local, FG, BG

# The knobs are easy. Just one continuous value per knob, [ knobs ]. So it should go in the state.
# An index is a vertex of a hypercube. Hypercubes!!!!!!


class FakeIndexSpace(spaces.Discrete):
    # Historically, most people use an index space that looks like this.
    # You need to hardcode every single index that it will consider, and it is 0/1 whether the
    # index is used or not.
    pass


class IndexSpace(spaces.Tuple):
    def __init__(
            self,
            gym_spec: GymSpec,
            seed: int = 15721,
    ):
        # An index is represented as (schema) x (order),
        # where 1 selects the active attributes out of the schema and the order provides the ordering.
        # TODO(WAN): better explanation of the hypercube-ish representation.
        self._gym_spec = gym_spec
        self._rng = np.random.default_rng(seed=seed)

        # Define the schema space.
        index_schema_space = spaces.Dict({
            schema_name: spaces.Dict({
                table_name: spaces.Dict({
                    column["name"]: spaces.Discrete(2, seed=seed)
                    for column in gym_spec.snapshot["schemas"][schema_name][table_name]["columns"]
                }, seed=seed)
                for table_name in gym_spec.snapshot["schemas"][schema_name]
            }, seed=seed)
            for schema_name in gym_spec.snapshot["schemas"]
        }, seed=seed)
        # Define the order space.
        # TODO(WAN):
        #   I think it may make sense to impose a cap on the number of attributes that
        #   will be indexed, say 5, so that this doesn't get blown up by some ridiculous table.
        max_num_attributes = max([len(attrs["columns"]) for _, _, attrs in self._gym_spec.schema_summary])
        index_order_space = spaces.MultiDiscrete([max_num_attributes] * max_num_attributes, seed=seed)

        super().__init__(spaces=[index_schema_space, index_order_space], seed=seed)

    def sample(self, schema_mask: Optional[np.ndarray] = None, order_value: Optional[np.ndarray] = None) -> tuple:
        # First, pick a schema and a table to explore indexes on, if none were specified.
        # Note that this may end up picking nothing -- very low chance of this happening though.
        if schema_mask is None:
            schema_name, table_name, _ = random.choice(self._gym_spec.schema_summary)
            schema_mask = self.make_schema_mask(schema_name=schema_name, table_name=table_name)
        # Next, instantiate an index to sample by picking attributes out of a schema.
        schema = self.spaces[0].sample(schema_mask)
        # Then, generate an ordering for the picked schema attributes.
        order_value = order_value if order_value is not None else self.make_order_value(schema)
        return schema, order_value

    def make_schema_mask(self, schema_name=None, table_name=None, column_name=None):
        gym_spec = self._gym_spec
        mask = {}
        for _schema_name in gym_spec.snapshot["schemas"]:
            mask[_schema_name] = {}
            for _table_name in gym_spec.snapshot["schemas"][_schema_name]:
                mask[_schema_name][_table_name] = {}
                for _column in gym_spec.snapshot["schemas"][_schema_name][_table_name]["columns"]:
                    matches_schema = schema_name is None or (schema_name is not None and _schema_name == schema_name)
                    matches_table = table_name is None or (table_name is not None and _table_name == table_name)
                    matches_column = column_name is None or (column_name is not None and _column["name"] == column_name)
                    action_possible = matches_schema and matches_table and matches_column
                    mask[_schema_name][_table_name][_column["name"]] = np.array([1, action_possible], dtype=np.int8)
        return mask

    def make_order_value(self, schema):
        """

        Parameters
        ----------
        schema
            An instantiated schema.

        Returns
        -------

        """
        # Count the number of active attributes in the schema.
        schema_active = [
            schema[schema_name][table_name][column_name]
            for schema_name in schema
            for table_name in schema[schema_name]
            for column_name in schema[schema_name][table_name]
        ]
        num_active = sum(schema_active)
        # Generate a permutation of the active attributes.
        permutation = np.arange(1, num_active + 1)
        self._rng.shuffle(permutation)
        # Then force this permutation.
        active_indexes = np.nonzero(schema_active)
        for idx, order in zip(*active_indexes, permutation):
            schema_active[idx] = order
        return np.array(schema_active, dtype=np.int8)


class KnobSpace(spaces.Tuple):
    pass


class QPPNetFeatures(spaces.Dict):
    """
    The table below is based on Table 2: QPP Net Inputs from
    Plan-Structured Deep Neural Network Models for Query Performance Prediction
    Ryan Marcus, Olga Papaemmanouil
    https://arxiv.org/abs/1902.00132

    However, there have since been updates to PostgreSQL. Use this file to figure out the current valid options.
    https://github.com/postgres/postgres/blob/master/src/backend/commands/explain.c
    Changed lines are marked with ! and small typo/wording changes are made where I see fit.

        Feature             PostgreSQL ops  Encoding    Description
        Plan Width          All             Numeric     Optimizer's estimate of the width of each output row
        Plan Rows           All             Numeric     Optimizer's estimate of the cardinality of the output
                                                        of the operator
        Plan Buffers        All             Numeric     Optimizer's estimate of the memory requirements
                                                        of an operator
        Estimated I/Os      All             Numeric     Optimizer's estimate of the number of I/Os performed
        Total Cost          All             Numeric     Optimizer's cost estimate for this operator, plus the subtree
        Join Type           Joins           One-hot     ! One of: Inner, Left, Full, Right, Semi, Anti
        Parent Relationship Joins           One-hot     When the child of a join.
                                                        ! One of: Inner, Outer, Subquery, Member, children, child
        Hash Buckets        Hash            Numeric     # hash buckets for hashing
        Hash Algorithm      Hash            One-hot     Hashing algorithm used
        Sort Key            Sort            One-hot     Key for sort operator
        Sort Method         Sort            One-hot     Sorting algorithm, e.g. "quicksort", "top-N heapsort",
                                                        "external sort"
                                                        ! One of: still in progress, top-N heapsort, quicksort,
                                                                  external sort, external merge
        Relation Name       All Scans       One-hot     Base relation of the leaf
        Attribute Mins      All Scans       Numeric     Vector of minimum values for relevant attributes
        Attribute Medians   All Scans       Numeric     Vector of median values for relevant attributes
        Attribute Maxs      All Scans       Numeric     Vector of maximum values for relevant attributes
        Index Name          Index Scans     One-hot     Name of index
        Scan Direction      Index Scans     ! One-hot   Direction to read the index (Backward, NoMovement, Forward)
        Strategy            Aggregate       One-hot     ! One of: plain, sorted, hashed, mixed
        Partial Mode        Aggregate       Boolean     Eligible to participate in parallel aggregation
        Operation           Aggregate       One-hot     The aggregation to perform, e.g. max, min, avg

    """

    def __init__(
            self,
            gym_spec: GymSpec,
            seed: int = 15721,
    ):
        assert len(gym_spec.snapshot["schemas"]) == 1, "No support for multiple schemas."
        self._gym_spec = gym_spec
        self._relations = [table_name for _, table_name, _ in self._gym_spec.schema_summary]
        self._indexes = [(table_name, index)
                         for _, table_name, attrs in self._gym_spec.schema_summary
                         for index in attrs["indexes"]]
        self._max_num_attributes = max([len(attrs["columns"]) for _, _, attrs in self._gym_spec.schema_summary])
        self._num_relations = len(self._relations)
        self._num_indexes = len(self._indexes)

        space_dict = {
            "Plan Width": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Plan Rows": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Plan Buffers": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Estimated I/Os": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Total Cost": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Join Type": spaces.Discrete(7, seed=seed),  # invalid, Inner, Left, Full, Right, Semi, Anti
            # invalid, Inner, Outer, Subquery, Member, children, child
            "Parent Relationship": spaces.Discrete(7, seed=seed),
            "Hash Buckets": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            # TODO(WAN):
            #  I don't know what Hash Algorithm refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Hash Algorithm": ,
            "Sort Key": spaces.MultiBinary(self._num_relations * self._max_num_attributes, seed=seed),
            # invalid, still in progress, top-N heapsort, quicksort, external sort, external merge
            "Sort Method": spaces.Discrete(6, seed=seed),
            "Relation Name": spaces.MultiBinary(self._num_relations, seed=seed),
            "Attribute Mins": spaces.Sequence(spaces.Box(low=-np.inf, high=np.inf, seed=seed), seed=seed),
            "Attribute Medians": spaces.Sequence(spaces.Box(low=-np.inf, high=np.inf, seed=seed), seed=seed),
            "Attribute Maxs": spaces.Sequence(spaces.Box(low=-np.inf, high=np.inf, seed=seed), seed=seed),
            "Index Name": spaces.MultiBinary(self._num_indexes, seed=seed),
            "Scan Direction": spaces.Discrete(3, seed=seed),  # invalid, Backward, NoMovement, Forward
            "Strategy": spaces.Discrete(5, seed=seed),  # invalid, plain, sorted, hashed, mixed
            "Partial Mode": spaces.Discrete(3, seed=seed),  # invalid, off, on
            # TODO(WAN):
            #  I don't know what Operation (formerly Operator) refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Operation": ,
        }
        super().__init__(spaces=space_dict, seed=seed)
    #
    # def sample(self) -> dict:
    #     raise NotImplementedError("Sampling this doesn't make sense. Future me problem.")


class DbGymEnv(gym.Env):
    def __init__(self, gym_spec: GymSpec, seed=15721):
        self._rng = np.random.default_rng(seed=seed)
        self._gym_spec = gym_spec
        assert len(gym_spec.snapshot) == 1, "We only support one schema right now."
        self.action_space = IndexSpace(gym_spec=gym_spec, seed=seed)
        self.observation_space = QPPNetFeatures(gym_spec=gym_spec, seed=seed)
        self._trainer = PostgresTrainer(gym_spec=self._gym_spec, seed=seed)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        # Reset the RNG.
        self._rng = np.random.default_rng(seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self._trainer.delete_target_dbms()
        self._trainer.create_target_dbms()
        observation, info = self._run_workload()
        return observation, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # Play through the entire workload.
        observation, info = self._run_workload()
        reward = 0.0
        terminated, truncated = False, False

        # I need to be able to control individual SQL queries to tag them with explain, for example.

        return observation, reward, terminated, truncated, info

    def _setup_target_DBMS(self):
        pass

    def _run_workload(self) -> Tuple[ObsType, dict]:
        engine = create_engine(
            self._trainer.get_target_dbms_connstr_sqlalchemy(),
            pool_size=psutil.cpu_count(logical=False)
        )



        raise NotImplementedError
        observation = None
        info = {}
        return observation, info
