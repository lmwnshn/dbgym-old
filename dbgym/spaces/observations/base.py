from abc import ABC
from gym import spaces


class BaseFeatureSpace(spaces.Space, ABC):
    SQL_PREFIX = "EXPLAIN (ANALYZE, FORMAT JSON, VERBOSE) "
