from abc import ABC

from gymnasium import spaces


class BaseFeatureSpace(spaces.Space, ABC):
    """
    All observations should inherit from BaseFeatureSpace.

    TODO(WAN):
      I think going down the rabbit hole of parsing SQL is too much work for too little gain.
      But obviously the prefix approach is fundamentally flawed, you need a full SQL deparser.

    Attributes
    ----------
    SQL_PREFIX: str
        The SQL prefix that needs to be prepended to every SQL query in order for the appropriate features to be
        generated.
    """

    SQL_PREFIX = "EXPLAIN (ANALYZE, FORMAT JSON, VERBOSE) "
