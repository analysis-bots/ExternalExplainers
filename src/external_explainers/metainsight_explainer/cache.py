from singleton_decorator import singleton
from collections import OrderedDict

PATTERN_CACHE_MAX_SIZE = 40000
DATASCOPE_CACHE_MAX_SIZE = 40000
PATTERN_EVAL_CACHE_MAX_SIZE = 40000
GROUPBY_CACHE_MAX_SIZE = 5000

@singleton
class Cache:
    """
    A singleton class to hold various caches used in the MetaInsight explainer.
    This helps in avoiding redundant computations and speeds up the evaluation process.
    We use a singleton pattern to make the cache:
    1. Global across the application.
    2. Persistent throughout the lifetime of the application.
    This cache is a simple LRU (Least Recently Used) cache implementation, removing the least recently used items when the cache exceeds its maximum size.
    The caches in this class are:
    - pattern_cache: Stores the data pattern objects evaluated for different data scopes and patterns.
    - datascope_cache: Stores the scores for different data scopes.
    - groupby_cache: Stores the results of groupby operations.
    - pattern_eval_cache: Stores the results of pattern evaluations on series.
    """

    def __init__(self):
        self._pattern_cache = OrderedDict()
        self._datascope_cache = OrderedDict()
        self._groupby_cache = OrderedDict()
        self._pattern_eval_cache = OrderedDict()
        self.pattern_cache_max_size = PATTERN_CACHE_MAX_SIZE
        self.datascope_cache_max_size = DATASCOPE_CACHE_MAX_SIZE
        self.groupby_cache_max_size = GROUPBY_CACHE_MAX_SIZE
        self.pattern_eval_cache_max_size = PATTERN_EVAL_CACHE_MAX_SIZE


    def _add_to_cache(self, cache, key, value, max_size) -> None:
        """
        Adds a key-value pair to the specified cache.
        If the cache exceeds its maximum size, it removes the least recently used item.
        """
        if key in cache:
            # Update the value and mark as recently used
            cache.move_to_end(key)
        cache[key] = value
        if len(cache) > max_size:
            # Pop the first item (least recently used)
            cache.popitem(last=False)


    def _get_from_cache(self, cache, key) -> any:
        """
        Retrieves a value from the specified cache by key.
        If the key exists, it marks the key as recently used.
        """
        if key in cache:
            # Move the accessed item to the end to mark it as recently used
            cache.move_to_end(key)
            return cache[key]
        return None


    def add_to_pattern_cache(self, key, value) -> None:
        """
        Adds a key-value pair to the pattern cache.
        If the cache exceeds its maximum size, it removes the least recently used item.
        """
        self._add_to_cache(self._pattern_cache, key, value, PATTERN_CACHE_MAX_SIZE)


    def add_to_datascope_cache(self, key, value) -> None:
        """
        Adds a key-value pair to the datascope cache.
        If the cache exceeds its maximum size, it removes the least recently used item.
        """
        self._add_to_cache(self._datascope_cache, key, value, DATASCOPE_CACHE_MAX_SIZE)

    def add_to_groupby_cache(self, key, value):
        """
        Adds a key-value pair to the groupby cache.
        If the cache exceeds its maximum size, it removes the least recently used item.
        """
        self._add_to_cache(self._groupby_cache, key, value, GROUPBY_CACHE_MAX_SIZE)

    def add_to_pattern_eval_cache(self, key, value) -> None:
        """
        Adds a key-value pair to the pattern evaluation cache.
        If the cache exceeds its maximum size, it removes the least recently used item.
        """
        self._add_to_cache(self._pattern_eval_cache, key, value, PATTERN_EVAL_CACHE_MAX_SIZE)


    def get_from_pattern_cache(self, key):
        """
        Retrieves a value from the pattern cache by key.
        If the key exists, it marks the key as recently used.
        """
        return self._get_from_cache(self._pattern_cache, key)

    def get_from_datascope_cache(self, key):
        """
        Retrieves a value from the datascope cache by key.
        If the key exists, it marks the key as recently used.
        """
        return self._get_from_cache(self._datascope_cache, key)

    def get_from_groupby_cache(self, key):
        """
        Retrieves a value from the groupby cache by key.
        If the key exists, it marks the key as recently used.
        """
        return self._get_from_cache(self._groupby_cache, key)

    def get_from_pattern_eval_cache(self, key):
        """
        Retrieves a value from the pattern evaluation cache by key.
        If the key exists, it marks the key as recently used.
        """
        return self._get_from_cache(self._pattern_eval_cache, key)