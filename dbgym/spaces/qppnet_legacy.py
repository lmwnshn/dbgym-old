from collections import OrderedDict
from typing import Any, Optional

import numpy as np
from gym import spaces

from dbgym.envs.gym_spec import GymSpec


class QPPNetFeatures(spaces.Sequence):
    """
    The table below is based on Table 2: QPP Net Inputs from
    Plan-Structured Deep Neural Network Models for Query Performance Prediction
    Ryan Marcus, Olga Papaemmanouil
    https://arxiv.org/abs/1902.00132

    However, there have since been updates to PostgreSQL. Use this file to figure out the current valid options.
    https://github.com/postgres/postgres/blob/master/src/backend/commands/explain.c
    Changed lines are marked with ! and small typo/wording changes are made where I see fit.

        Feature             PostgreSQL ops  Encoding    Description
        Observation Index   All             Numeric     The index of this observation.
        ! Children Indexes  All             [Numeric]   List of observation indexes for direct children nodes.
        ! Query Num         All             Numeric     The number of this query, used to associate datapoints together.
        ! Node Position     All             [Numeric]   Node location.
                                                        [0] = root; [0,0] = root,child0; [0,1] = root,child1; etc.
        ! Query Hash        All             [Numeric]   Query hash, used to associate datapoints together.
        ! Node Type         All             One-hot     The node type, added because we flatten to a common format.
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
        Partial Mode        Aggregate       ! One-hot   Eligible to participate in parallel aggregation
                                                        One of: Simple, Partial, Finalize
        Operation           Aggregate       One-hot     The aggregation to perform, e.g. max, min, avg

    """

    def __init__(
        self,
        gym_spec: GymSpec,
        seed: int = 15721,
    ):
        assert (
            len(gym_spec.snapshot["schemas"]) == 1
        ), "No support for multiple schemas."
        self._gym_spec = gym_spec
        self._relations = [
            table_name for _, table_name, _ in self._gym_spec.schema_summary
        ]
        self._attribute_names = {
            table_name: [column["name"] for column in attributes["columns"]]
            for _, table_name, attributes in self._gym_spec.schema_summary
        }
        self._attribute_stats = {
            table_name: {
                column["name"]: column["stats"] for column in attributes["columns"]
            }
            for _, table_name, attributes in self._gym_spec.schema_summary
        }
        self._indexes = [
            (table_name, index["name"])
            for _, table_name, attrs in self._gym_spec.schema_summary
            for index in attrs["indexes"]
        ]
        self._max_num_attributes = max(
            [len(attrs["columns"]) for _, _, attrs in self._gym_spec.schema_summary]
        )
        self._num_relations = len(self._relations)
        self._num_indexes = len(self._indexes)

        self._node_types = [
            "Aggregate",
            "Append",
            "Bitmap Heap Scan",
            "Bitmap Index Scan",
            "Function Scan",
            "Hash",
            "Hash Join",
            "Index Scan",
            "Index Only Scan",
            "Limit",
            "LockRows",
            "Memoize",
            "Merge Join",
            "ModifyTable",
            "Nested Loop",
            "ProjectSet",
            "Result",
            "Seq Scan",
            "Subquery Scan",
            "Sort",
            "WindowAgg",
        ]
        self._join_types = ["invalid", "Inner", "Left", "Full", "Right", "Semi", "Anti"]
        self._parent_relationships = [
            "invalid",
            "Inner",
            "Outer",
            "Subquery",
            "Member",
            "children",
            "child",
        ]
        self._sort_methods = [
            "invalid",
            "still in progress",
            "top-N heapsort",
            "quicksort",
            "external sort",
            "external merge",
        ]
        self._scan_directions = ["invalid", "Backward", "NoMovement", "Forward"]
        self._strategies = ["invalid", "Plain", "Sorted", "Hashed", "Mixed"]
        self._partial_modes = ["invalid", "Simple", "Partial", "Finalize"]

        space_dict = {
            "Observation Index": spaces.Box(
                low=0, high=np.inf, dtype=np.int64, seed=seed
            ),
            "Children Indexes": spaces.Sequence(
                spaces.Box(low=0, high=np.inf, dtype=np.int64, seed=seed), seed=seed
            ),
            # We add Query Num and Node Position so that we can flatten out our observations.
            "Query Num": spaces.Box(low=0, high=np.inf, dtype=np.int32, seed=seed),
            "Node Position": spaces.Sequence(
                spaces.Box(low=0, high=np.inf, dtype=np.int32, seed=seed), seed=seed
            ),
            # We add Query Hash to group observations later.
            "Query Hash": spaces.Sequence(
                spaces.Discrete(len(self._node_types), seed=seed), seed=seed
            ),
            # We add "Node Type" because we have flattened all nodes into a common representation.
            "Node Type": spaces.MultiBinary(len(self._node_types), seed=seed),
            "Plan Width": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Plan Rows": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            # TODO(WAN):
            #  I don't know what Plan Buffers refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Plan Buffers": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            # TODO(WAN):
            #  I don't know what Estimated I/Os refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Estimated I/Os": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Total Cost": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Join Type": spaces.MultiBinary(len(self._join_types), seed=seed),
            "Parent Relationship": spaces.MultiBinary(
                len(self._parent_relationships), seed=seed
            ),
            "Hash Buckets": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            # TODO(WAN):
            #  I don't know what Hash Algorithm refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Hash Algorithm": ,
            "Sort Key": spaces.MultiBinary(
                self._num_relations * self._max_num_attributes, seed=seed
            ),
            "Sort Method": spaces.MultiBinary(len(self._sort_methods), seed=seed),
            "Relation Name": spaces.MultiBinary(self._num_relations, seed=seed),
            "Attribute Mins": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._max_num_attributes,), seed=seed
            ),
            "Attribute Medians": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._max_num_attributes,), seed=seed
            ),
            "Attribute Maxs": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._max_num_attributes,), seed=seed
            ),
            "Index Name": spaces.MultiBinary(self._num_indexes, seed=seed),
            "Scan Direction": spaces.MultiBinary(len(self._scan_directions), seed=seed),
            "Strategy": spaces.MultiBinary(len(self._strategies), seed=seed),
            "Partial Mode": spaces.MultiBinary(len(self._partial_modes), seed=seed),
            # TODO(WAN):
            #  I don't know what Operation (formerly Operator) refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Operation": ,
            "Actual Total Time": spaces.Box(
                low=0, high=np.inf, dtype=np.float32, seed=seed
            ),
        }
        explain_space = spaces.Dict(spaces=space_dict, seed=seed)
        super().__init__(space=explain_space, seed=seed)

    def sample(self, mask: Optional[dict[str, Any]] = None) -> dict:
        raise NotImplementedError(
            "Sampling this doesn't make sense. Future me problem."
        )

    def generate(self, result_dict, query_num, observation_idx) -> list:
        plan_dict = result_dict["Plan"]
        query_hash = self._compute_hash(plan_dict)
        node_position = [0]
        return self._generate(
            plan_dict, query_num, query_hash, node_position, observation_idx
        )

    def _compute_hash(self, plan_dict):
        # hash(qp) = enc(qp["Node Type"]) || (hash(child) for child in qp["Plans"])
        node_type = self._node_types.index(plan_dict["Node Type"])
        res = [node_type]
        if "Plans" in plan_dict:
            for child in plan_dict["Plans"]:
                res.extend(self._compute_hash(child))
        return res

    def _generate(
        self, plan_dict, query_num, query_hash, node_position, observation_idx
    ) -> list:
        def _singleton(x, dtype=np.float32):
            return np.array([x], dtype=dtype)

        def _one_hot(valid_choices, key):
            arr = [0 for _ in valid_choices]
            if key in plan_dict:
                val = plan_dict[key]
                assert (
                    val in valid_choices
                ), f"What's this? {valid_choices} {plan_dict} {key}"
                val_idx = valid_choices.index(plan_dict[key])
            else:
                assert "invalid" in valid_choices
                val_idx = valid_choices.index("invalid")
            arr[val_idx] = 1
            return np.array(arr, dtype=np.int8)

        observations = []
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
                            table_index = (
                                self._relations.index(rel_name)
                                * self._max_num_attributes
                            )
                            attr_index = self._attribute_names[rel_name].index(
                                attr_name
                            )
                            sort_key[table_index + attr_index] = 1

        sort_key = np.array(sort_key, dtype=np.int8)

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
                        output_index_name[
                            self._indexes.index((rel_name, index_name))
                        ] = 1

        output_relation_name = np.array(output_relation_name, dtype=np.int8)
        output_index_name = np.array(output_index_name, dtype=np.int8)

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
                            # TODO(WAN): string support? We didn't handle this before either AFAIK.
                            if (
                                type(self._attribute_stats[rel_name][attr_name]["min"])
                                == str
                            ):
                                continue
                            output_min_vec[idx] = self._attribute_stats[rel_name][
                                attr_name
                            ]["min"]
                            output_med_vec[idx] = self._attribute_stats[rel_name][
                                attr_name
                            ]["median"]
                            output_max_vec[idx] = self._attribute_stats[rel_name][
                                attr_name
                            ]["max"]

        output_min_vec = np.array(output_min_vec, dtype=np.float32)
        output_med_vec = np.array(output_med_vec, dtype=np.float32)
        output_max_vec = np.array(output_max_vec, dtype=np.float32)

        output_observation_index = _singleton(observation_idx, dtype=np.int64)
        observation_idx += 1
        children_indexes = []
        if "Plans" in plan_dict:
            for i, plan in enumerate(plan_dict["Plans"]):
                new_position = node_position + [i]
                children_indexes.append(observation_idx)
                child_observations = self._generate(
                    plan, query_num, query_hash, new_position, observation_idx
                )
                observations.extend(child_observations)
                observation_idx += len(child_observations)
        output_children_indexes = [
            _singleton(idx, dtype=np.int64) for idx in children_indexes
        ]

        output_node_position = [_singleton(x, dtype=np.int32) for x in node_position]

        # Force the ordering to match sample() output.
        ordered_dict_items = [
            ("Actual Total Time", _singleton(plan_dict["Actual Total Time"])),
            ("Attribute Maxs", output_max_vec),
            ("Attribute Medians", output_med_vec),
            ("Attribute Mins", output_min_vec),
            ("Children Indexes", output_children_indexes),
            (
                "Hash Buckets",
                _singleton(
                    plan_dict["Hash Buckets"] if "Hash Buckets" in plan_dict else 0
                ),
            ),
            ("Index Name", output_index_name),
            ("Join Type", _one_hot(self._join_types, "Join Type")),
            ("Node Position", output_node_position),
            ("Node Type", _one_hot(self._node_types, "Node Type")),
            ("Observation Index", output_observation_index),
            (
                "Parent Relationship",
                _one_hot(self._parent_relationships, "Parent Relationship"),
            ),
            ("Partial Mode", _one_hot(self._partial_modes, "Partial Mode")),
            ("Plan Rows", _singleton(plan_dict["Plan Rows"])),
            ("Plan Width", _singleton(plan_dict["Plan Width"])),
            ("Query Hash", query_hash),
            ("Query Num", _singleton(query_num, dtype=np.int32)),
            ("Relation Name", output_relation_name),
            ("Scan Direction", _one_hot(self._scan_directions, "Scan Direction")),
            ("Sort Key", sort_key),
            ("Sort Method", _one_hot(self._sort_methods, "Sort Method")),
            ("Strategy", _one_hot(self._strategies, "Strategy")),
            ("Total Cost", _singleton(plan_dict["Total Cost"])),
            # TODO(WAN):
            #  Where are these from? Who knows?
            # "Plan Buffers": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            # "Estimated I/Os": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            # TODO(WAN):
            #  I don't know what Hash Algorithm refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Hash Algorithm": ,
            # TODO(WAN):
            #  I don't know what Operation (formerly Operator) refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Operation":
        ]
        observations.append(OrderedDict(ordered_dict_items))

        return observations
