import numpy as np
import pandas as pd
from gym import spaces

from dbgym.envs.gym_spec import GymSpec
from collections import OrderedDict

class GpredictorFeatures:
    _node_types = [
        "Aggregate",
        "Append",
        "BitmapAnd",
        "BitmapOr",
        "Bitmap Heap Scan",
        "Bitmap Index Scan",
        "CTE Scan",
        "Custom Scan",
        "Foreign Scan",
        "Function Scan",
        "Gather",
        "Gather Merge",
        "Group",
        "Hash",
        "Hash Join",
        "Incremental Sort",
        "Index Scan",
        "Index Only Scan",
        "Limit",
        "LockRows",
        "Materialize",
        "Memoize",
        "Merge Append",
        "Merge Join",
        "ModifyTable",
        "Named Tuplestore Scan",
        "Nested Loop",
        "ProjectSet",
        "Recursive Union",
        "Result",
        "Sample Scan",
        "Seq Scan",
        "SetOp",
        "Subquery Scan",
        "Sort",
        "Table Function Scan",
        "Tid Scan",
        "Tid Range Scan",
        "Unique",
        "WindowAgg",
        "WorkTable Scan",
        "Values Scan",
    ]

    modify_op_map = {'Update': 1, 'Insert': 2, 'Delete': 3}

    def __init__(
        self,
        gym_spec: GymSpec,
        seed: int = 15721,
        batch: int = 64,
        
    ):
        assert len(gym_spec.snapshot["schemas"]) == 1, "No support for multiple schemas."
        self._gym_spec = gym_spec
        self._relation_name_map = dict()
        for _, table_name, _ in self._gym_spec.schema_summary:
            self._relation_name_map[table_name] = len(self._relation_name_map)

        self._num_relations = len(self._relation_name_map)
        self._batch = batch

        space_dict = {
            "Actual Total Time (us)": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Features": spaces.Sequence(
                spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, seed=seed),
                seed=seed,
            ),
            "Neighbors": spaces.Sequence(
                spaces.Box(low=0, high=np.inf, dtype=np.int64, seed=seed),
                seed=seed,
            ),
            "Weights": spaces.Sequence(
                spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, seed=seed),
                seed=seed,
            ),
            "Node Type": spaces.MultiBinary(len(self._node_types), seed=seed),
            "Observation Index": spaces.Box(low=0, high=np.inf, dtype=np.int64, seed=seed),
            "Query Hash": spaces.Sequence(spaces.Discrete(len(self._node_types), seed=seed), seed=seed),
            "Query Num": spaces.Box(low=0, high=np.inf, dtype=np.int32, seed=seed),
            "Query Text": spaces.Sequence(
                spaces.Box(low=-np.inf, high=np.inf, dtype=np.char, seed=seed),
                seed=seed,
            ),
            "Is Query Root": spaces.Box(low=False, high=True, dtype=np.bool8, seed=seed),
            "Is Batch End": spaces.Box(low=False, high=True, dtype=np.bool8, seed=seed),
        }

    def generate(self, result_dict, query_num, observation_idx, curr_batch_base_ou_idx, graph_idx_map) -> list:
        plan_dict = result_dict["Plan"]
        query_hash = self._featurize_query_hash(plan_dict)
        return self._generate(plan_dict, query_num, query_hash, observation_idx, curr_batch_base_ou_idx, graph_idx_map)

    def _generate(self, plan_dict, query_num, query_hash, observation_idx) -> list:
        observations = []

        # Generate children observations.
        output_observation_index = self._singleton(observation_idx, dtype=np.int64)
        children_observation_indexes = []
        observation_idx += 1
        if "Plans" in plan_dict:
            for i, plan in enumerate(plan_dict["Plans"]):
                children_observation_indexes.append(observation_idx)
                child_observations = self._generate(plan, query_num, query_hash, observation_idx)
                observations.extend(child_observations)
                observation_idx += len(child_observations)
        output_children_observation_indexes = [
            self._singleton(idx, dtype=np.int64) for idx in children_observation_indexes
        ]

        ordered_dict_items = [
            ("Actual Total Time (ms)", self._singleton(plan_dict["Actual Total Time"])),
            ("Children Observation Indexes", output_children_observation_indexes),
            ("Features", self._featurize(plan_dict)),
            ("Node Type", self._one_hot(self._node_types, plan_dict, "Node Type")),
            ("Observation Index", output_observation_index),
            ("Query Hash", query_hash),
            ("Query Num", self._singleton(query_num, dtype=np.int32)),
        ]
        observations.append(OrderedDict(ordered_dict_items))
        return observations

    
    def batch(self):
        return self._batch
    pass