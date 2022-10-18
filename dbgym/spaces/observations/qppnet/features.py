import datetime
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import pandas as pd
from gym import spaces
from sklearn.preprocessing import MinMaxScaler

from dbgym.envs.gym_spec import GymSpec


class QPPNetFeatures(spaces.Sequence):
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
    _join_types = ["invalid", "Inner", "Left", "Full", "Right", "Semi", "Anti"]
    _parent_relationships = [
        "invalid",
        "Inner",
        "Outer",
        "Subquery",
        "Member",
        "children",
        "child",
    ]
    _sort_methods = [
        "invalid",
        "still in progress",
        "top-N heapsort",
        "quicksort",
        "external sort",
        "external merge",
    ]
    _scan_directions = ["invalid", "Backward", "NoMovement", "Forward"]
    _strategies = ["invalid", "Plain", "Sorted", "Hashed", "Mixed"]
    _partial_modes = ["invalid", "Simple", "Partial", "Finalize"]

    def __init__(
        self,
        gym_spec: GymSpec,
        seed: int = 15721,
    ):
        assert len(gym_spec.snapshot["schemas"]) == 1, "No support for multiple schemas."
        self._gym_spec = gym_spec
        self._relations = [table_name for _, table_name, _ in self._gym_spec.schema_summary]
        self._attribute_names = {
            table_name: [column["name"] for column in attributes["columns"]]
            for _, table_name, attributes in self._gym_spec.schema_summary
        }
        self._attribute_stats = {
            table_name: {column["name"]: column["stats"] for column in attributes["columns"]}
            for _, table_name, attributes in self._gym_spec.schema_summary
        }
        self._indexes = [
            (table_name, index["name"])
            for _, table_name, attrs in self._gym_spec.schema_summary
            for index in attrs["indexes"]
        ]
        self._max_num_attributes = max([len(attrs["columns"]) for _, _, attrs in self._gym_spec.schema_summary])
        self._num_relations = len(self._relations)
        self._num_indexes = len(self._indexes)

        space_dict = {
            "Actual Total Time (ms)": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Children Observation Indexes": spaces.Sequence(
                spaces.Box(low=0, high=np.inf, dtype=np.int64, seed=seed), seed=seed
            ),
            "Features": spaces.Sequence(
                spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, seed=seed),
                seed=seed,
            ),
            "Node Type": spaces.MultiBinary(len(self._node_types), seed=seed),
            "Observation Index": spaces.Box(low=0, high=np.inf, dtype=np.int64, seed=seed),
            "Query Hash": spaces.Sequence(spaces.Discrete(len(self._node_types), seed=seed), seed=seed),
            "Query Num": spaces.Box(low=0, high=np.inf, dtype=np.int32, seed=seed),
        }
        explain_space = spaces.Dict(spaces=space_dict, seed=seed)
        super().__init__(space=explain_space, seed=seed)

    def sample(self, mask: Optional[dict[str, Any]] = None) -> dict:
        raise NotImplementedError("Sampling this doesn't make sense. Future me problem.")

    def generate(self, result_dict, query_num, observation_idx) -> list:
        plan_dict = result_dict["Plan"]
        query_hash = self._featurize_query_hash(plan_dict)
        return self._generate(plan_dict, query_num, query_hash, observation_idx)

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

    def _featurize(self, plan_dict):
        node_type = plan_dict["Node Type"]
        features = []
        # Refer to Table 2: QPP Net Inputs.

        # All.
        features.append(plan_dict["Plan Width"])
        features.append(plan_dict["Plan Rows"])
        # TODO(WAN): plan buffers, estimated I/Os.
        features.append(plan_dict["Total Cost"])

        # Joins.
        if node_type in ["Hash Join", "Merge Join"]:
            features.extend(self._one_hot(self._join_types, plan_dict, "Join Type"))
            features.extend(self._one_hot(self._parent_relationships, plan_dict, "Parent Relationship"))
        # Hash.
        elif node_type in ["Hash"]:
            features.append(plan_dict["Hash Buckets"] if "Hash Buckets" in plan_dict else 0)
            # TODO(WAN): Hash Algorithm
        # Sort.
        elif node_type in ["Sort"]:
            features.extend(self._featurize_sort_key(plan_dict))
            features.extend(self._one_hot(self._sort_methods, plan_dict, "Sort Method"))
        # Scans.
        elif node_type in ["Seq Scan", "Bitmap Heap Scan"]:
            output_relation_name, _ = self._featurize_relation_name_index_name(plan_dict)
            (
                output_min_vec,
                output_med_vec,
                output_max_vec,
            ) = self._featurize_min_med_max(plan_dict)
            features.extend(output_relation_name)
            features.extend(output_min_vec)
            features.extend(output_med_vec)
            features.extend(output_max_vec)
        # Index scans.
        elif node_type in ["Index Scan", "Index Only Scan"]:
            (
                output_relation_name,
                output_index_name,
            ) = self._featurize_relation_name_index_name(plan_dict)
            (
                output_min_vec,
                output_med_vec,
                output_max_vec,
            ) = self._featurize_min_med_max(plan_dict)
            features.extend(output_relation_name)
            features.extend(output_min_vec)
            features.extend(output_med_vec)
            features.extend(output_max_vec)
            features.extend(output_index_name)
            features.extend(self._one_hot(self._scan_directions, plan_dict, "Scan Direction"))
        elif node_type in ["Bitmap Index Scan"]:
            _, output_index_name = self._featurize_relation_name_index_name(plan_dict)
            features.extend(output_index_name)
        # Aggregates.
        elif node_type in ["Aggregate"]:
            features.extend(self._one_hot(self._strategies, plan_dict, "Strategy"))
            features.extend(self._one_hot(self._partial_modes, plan_dict, "Partial Mode"))
        # TODO(WAN): Operator

        return [self._singleton(x) for x in features]

    def _featurize_min_med_max(self, plan_dict):
        output_min_vec = [0] * self._max_num_attributes
        output_med_vec = [0] * self._max_num_attributes
        output_max_vec = [0] * self._max_num_attributes
        if "Relation Name" in plan_dict:
            target = None
            if "Filter" in plan_dict:
                target = "Filter"
            elif "Recheck Cond" in plan_dict:
                target = "Recheck Cond"
            if target is not None:
                target = plan_dict[target]
                rel_name = plan_dict["Relation Name"]
                if not rel_name.startswith("pg_"):
                    attributes = self._attribute_names[rel_name]
                    for idx, attr_name in enumerate(attributes):
                        if attr_name in target:
                            min_val, med_val, max_val = [
                                self._attribute_stats[rel_name][attr_name][x] for x in ["min", "median", "max"]
                            ]
                            # TODO(WAN): string support? We didn't handle this before either AFAIK.
                            if type(min_val) == str:
                                continue
                            if type(min_val) in [datetime.datetime, datetime.date]:
                                min_val = pd.Timestamp(min_val).timestamp()
                                med_val = pd.Timestamp(med_val).timestamp()
                                max_val = pd.Timestamp(max_val).timestamp()
                            output_min_vec[idx] = min_val
                            output_med_vec[idx] = med_val
                            output_max_vec[idx] = max_val
        return output_min_vec, output_med_vec, output_max_vec

    def _featurize_query_hash(self, plan_dict):
        # hash(qp) = enc(qp["Node Type"]) || (hash(child) for child in qp["Plans"])
        node_type = self._node_types.index(plan_dict["Node Type"])
        output_hash = [node_type]
        if "Plans" in plan_dict:
            for child in plan_dict["Plans"]:
                output_hash.extend(self._featurize_query_hash(child))
        return output_hash

    def _featurize_relation_name_index_name(self, plan_dict):
        output_relation_name = [0] * self._num_relations
        output_index_name = [0] * self._num_indexes
        if "Relation Name" in plan_dict:
            # TODO(WAN): Copied over behavior from our QPPNet; ignores pg_ tables.
            rel_name = plan_dict["Relation Name"]
            if not rel_name.startswith("pg_"):
                output_relation_name[self._relations.index(rel_name)] = 1
                if "Index Name" in plan_dict:
                    index_name = plan_dict["Index Name"]
                    if not index_name.startswith("pg_"):
                        output_index_name[self._indexes.index((rel_name, index_name))] = 1
        return output_relation_name, output_index_name

    def _featurize_sort_key(self, plan_dict):
        # Generate sort key (not always valid).
        # Each relation is padded to have the maximum number of attributes,
        # and then the sort keys in the input plan are set to 1.
        sort_key = [0] * (self._num_relations * self._max_num_attributes)
        if "Sort Key" in plan_dict:
            for key in plan_dict["Sort Key"]:
                # TODO(WAN): This code is brought over from old QPPNet. Might be wrong.
                key = key.replace("(", " ").replace(")", " ")
                for subkey in key.split(" "):
                    if subkey != " " and "." in subkey:
                        rel_name, attr_name = subkey.split(" ")[0].split(".")
                        if rel_name in self._relations:
                            table_index = self._relations.index(rel_name) * self._max_num_attributes
                            attr_index = self._attribute_names[rel_name].index(attr_name)
                            sort_key[table_index + attr_index] = 1
        return sort_key

    @staticmethod
    def _singleton(x, dtype=np.float32):
        return np.array([x], dtype=dtype)

    @staticmethod
    def _one_hot(valid_choices, plan_dict, key):
        arr = [0 for _ in valid_choices]
        if key in plan_dict:
            val = plan_dict[key]
            assert val in valid_choices, f"What's this? {valid_choices} {plan_dict} {key}"
            val_idx = valid_choices.index(plan_dict[key])
        else:
            assert "invalid" in valid_choices
            val_idx = valid_choices.index("invalid")
        arr[val_idx] = 1
        return np.array(arr, dtype=np.int8)

    @staticmethod
    def convert_observations_to_df(observations, convert_node_type=True, explode=False):
        def _is_listlike(item):
            return type(item) in [tuple, list, np.ndarray]

        # Convert the observations to a df.
        df = pd.json_normalize(observations)
        # Unbox values, e.g., [1] -> 1. This improves df compression and is more ergonomic.
        for col in df.columns:
            if _is_listlike(df[col].iloc[0]):
                lens = df[col].apply(len)
                if (lens == 1).all():
                    # If this is a Box shape, [x], then unwrap the ndarray value.
                    df[col] = df[col].apply(lambda arr: arr.item())
                elif len(df[col].iloc[lens.argmax()]) > 0 and _is_listlike(df[col].iloc[lens.argmax()]):
                    # Otherwise, this may be a Seq(Box) shape, [[x], [y]], so try flattening that.
                    df[col] = df[col].apply(lambda arr: np.array(arr).flatten())

        # Convert the one-hot node type encoding to the name of the node type.
        if convert_node_type:
            df["Node Type"] = df["Node Type"].apply(lambda onehot: QPPNetFeatures._node_types[onehot.argmax()])

        # Explode out arrays in columns.
        if explode:
            for col in df.columns:
                if _is_listlike(df[col].iloc[0]):
                    lens = df[col].apply(len)
                    # Find the maximum length and explode out.
                    max_len = lens.max()
                    df[col] = df[col].apply(lambda x: np.array(x).flatten())
                    if not (lens == max_len).all():
                        # Pad.
                        df[col] = df[col].apply(lambda arr: np.pad(arr, (0, max_len - arr.shape[0])))
                    new_cols = [f"{col}_{i}" for i in range(1, max_len + 1)]
                    df[new_cols] = pd.DataFrame(df[col].tolist(), index=df.index)
                    del df[col]

        # Sort the df.
        assert df["Observation Index"].nunique() == len(df), "Observation not unique?"
        df = df.sort_values("Observation Index").reset_index(drop=True)
        assert all(df["Observation Index"] == df.index), "Missing observations?"

        for cat in [
            "Query Hash",
        ]:
            df[cat] = df[cat].apply(tuple)  # .astype("category") # pending pyarrow support for nested dicts

        df["Actual Total Time (us)"] = df["Actual Total Time (ms)"] * 1000
        del df["Actual Total Time (ms)"]

        # Defragment the df.
        df = df.copy()
        return df

    @staticmethod
    def normalize_observations_df(observations, scalers:Optional[dict[str, MinMaxScaler]]=None):
        df = observations.copy()
        groups = df.groupby("Node Type")
        if scalers is None:
            scalers = {}
        for node_type, gdf in groups:
            feat_s = gdf["Features"]
            feat_df = pd.DataFrame.from_dict(dict(zip(feat_s.index, feat_s.values)), orient="index")
            if node_type in scalers:
                scaler = scalers[node_type]
                transformed_df = pd.DataFrame(scaler.transform(feat_df), index=feat_df.index)
            else:
                scaler = MinMaxScaler()
                transformed_df = pd.DataFrame(scaler.fit_transform(feat_df), index=feat_df.index)
                scalers[node_type] = scaler
            transformed_s = transformed_df.apply(lambda x: list(x), axis=1)
            df.loc[transformed_s.index, "Features"] = transformed_s

        node_type = "Fake_ActualTime"
        if node_type in scalers:
            scaler = scalers[node_type]
            df["Actual Total Time (scaled)"] = scaler.transform(df["Actual Total Time (us)"].values.reshape(-1, 1))
        else:
            scaler = MinMaxScaler()
            df["Actual Total Time (scaled)"] = scaler.fit_transform(df["Actual Total Time (us)"].values.reshape(-1, 1))
            scalers[node_type] = scaler

        return df, scalers
