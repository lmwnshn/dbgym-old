import random
from typing import Optional

import numpy as np
from gym import spaces

from dbgym.envs.gym_spec import GymSpec

# An index is a vertex of a hypercube. Hypercubes!!!!!!


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
        index_schema_space = spaces.Dict(
            {
                schema_name: spaces.Dict(
                    {
                        table_name: spaces.Dict(
                            {
                                column["name"]: spaces.Discrete(2, seed=seed)
                                for column in gym_spec.snapshot["schemas"][schema_name][table_name]["columns"]
                            },
                            seed=seed,
                        )
                        for table_name in gym_spec.snapshot["schemas"][schema_name]
                    },
                    seed=seed,
                )
                for schema_name in gym_spec.snapshot["schemas"]
            },
            seed=seed,
        )
        # Define the order space.
        # TODO(WAN):
        #   I think it may make sense to impose a cap on the number of attributes that
        #   will be indexed, say 5, so that this doesn't get blown up by some ridiculous table.
        max_num_attributes = max([len(attrs["columns"]) for _, _, attrs in self._gym_spec.schema_summary])
        index_order_space = spaces.MultiDiscrete([max_num_attributes] * max_num_attributes, seed=seed)

        super().__init__(spaces=[index_schema_space, index_order_space], seed=seed)

    def sample(
        self,
        schema_mask: Optional[np.ndarray] = None,
        order_value: Optional[np.ndarray] = None,
    ) -> tuple:
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
