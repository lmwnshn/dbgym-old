from gym import spaces
from dbgym.envs.gym_spec import GymSpec
import numpy as np

from typing import Any, Optional


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
        assert (
            len(gym_spec.snapshot["schemas"]) == 1
        ), "No support for multiple schemas."
        self._gym_spec = gym_spec
        self._relations = [
            table_name for _, table_name, _ in self._gym_spec.schema_summary
        ]
        self._indexes = [
            (table_name, index)
            for _, table_name, attrs in self._gym_spec.schema_summary
            for index in attrs["indexes"]
        ]
        self._max_num_attributes = max(
            [len(attrs["columns"]) for _, _, attrs in self._gym_spec.schema_summary]
        )
        self._num_relations = len(self._relations)
        self._num_indexes = len(self._indexes)

        space_dict = {
            "Plan Width": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Plan Rows": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Plan Buffers": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Estimated I/Os": spaces.Box(
                low=0, high=np.inf, dtype=np.float32, seed=seed
            ),
            "Total Cost": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Join Type": spaces.Discrete(
                7, seed=seed
            ),  # invalid, Inner, Left, Full, Right, Semi, Anti
            # invalid, Inner, Outer, Subquery, Member, children, child
            "Parent Relationship": spaces.Discrete(7, seed=seed),
            "Hash Buckets": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            # TODO(WAN):
            #  I don't know what Hash Algorithm refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Hash Algorithm": ,
            "Sort Key": spaces.MultiBinary(
                self._num_relations * self._max_num_attributes, seed=seed
            ),
            # invalid, still in progress, top-N heapsort, quicksort, external sort, external merge
            "Sort Method": spaces.Discrete(6, seed=seed),
            "Relation Name": spaces.MultiBinary(self._num_relations, seed=seed),
            "Attribute Mins": spaces.Sequence(
                spaces.Box(low=-np.inf, high=np.inf, seed=seed), seed=seed
            ),
            "Attribute Medians": spaces.Sequence(
                spaces.Box(low=-np.inf, high=np.inf, seed=seed), seed=seed
            ),
            "Attribute Maxs": spaces.Sequence(
                spaces.Box(low=-np.inf, high=np.inf, seed=seed), seed=seed
            ),
            "Index Name": spaces.MultiBinary(self._num_indexes, seed=seed),
            "Scan Direction": spaces.Discrete(
                3, seed=seed
            ),  # invalid, Backward, NoMovement, Forward
            "Strategy": spaces.Discrete(
                5, seed=seed
            ),  # invalid, plain, sorted, hashed, mixed
            "Partial Mode": spaces.Discrete(3, seed=seed),  # invalid, off, on
            # TODO(WAN):
            #  I don't know what Operation (formerly Operator) refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Operation": ,
        }
        super().__init__(spaces=space_dict, seed=seed)

    def sample(self, mask: Optional[dict[str, Any]] = None) -> dict:
        raise NotImplementedError(
            "Sampling this doesn't make sense. Future me problem."
        )


    def generate(self, explain):
        space_dict = {
            "Plan Width": explain["Plan Width"],
            "Plan Rows": explain["Plan Rows"],
            "Plan Buffers": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Estimated I/Os": spaces.Box(
                low=0, high=np.inf, dtype=np.float32, seed=seed
            ),
            "Total Cost": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            "Join Type": spaces.Discrete(
                7, seed=seed
            ),  # invalid, Inner, Left, Full, Right, Semi, Anti
            # invalid, Inner, Outer, Subquery, Member, children, child
            "Parent Relationship": spaces.Discrete(7, seed=seed),
            "Hash Buckets": spaces.Box(low=0, high=np.inf, dtype=np.float32, seed=seed),
            # TODO(WAN):
            #  I don't know what Hash Algorithm refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Hash Algorithm": ,
            "Sort Key": spaces.MultiBinary(
                self._num_relations * self._max_num_attributes, seed=seed
            ),
            # invalid, still in progress, top-N heapsort, quicksort, external sort, external merge
            "Sort Method": spaces.Discrete(6, seed=seed),
            "Relation Name": spaces.MultiBinary(self._num_relations, seed=seed),
            "Attribute Mins": spaces.Sequence(
                spaces.Box(low=-np.inf, high=np.inf, seed=seed), seed=seed
            ),
            "Attribute Medians": spaces.Sequence(
                spaces.Box(low=-np.inf, high=np.inf, seed=seed), seed=seed
            ),
            "Attribute Maxs": spaces.Sequence(
                spaces.Box(low=-np.inf, high=np.inf, seed=seed), seed=seed
            ),
            "Index Name": spaces.MultiBinary(self._num_indexes, seed=seed),
            "Scan Direction": spaces.Discrete(
                3, seed=seed
            ),  # invalid, Backward, NoMovement, Forward
            "Strategy": spaces.Discrete(
                5, seed=seed
            ),  # invalid, plain, sorted, hashed, mixed
            "Partial Mode": spaces.Discrete(3, seed=seed),  # invalid, off, on
            # TODO(WAN):
            #  I don't know what Operation (formerly Operator) refers to.
            #  The original paper has no code, our reimplementation didn't have it either.
            # "Operation": ,
        }

        for plan in explain["Plans"]:
            self.generate(plan)

