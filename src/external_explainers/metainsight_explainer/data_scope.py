import pandas as pd
from typing import Dict, List, Tuple
from scipy.special import kl_div
import re
from external_explainers.metainsight_explainer.cache import Cache

cache = Cache()

class DataScope:
    """
    A data scope, as defined in the MetaInsight paper.
    Contains 3 elements: subspace, breakdown and measure.
    Example: for the query SELECT Month, SUM(Sales) FROM DATASET WHERE City==“Los Angeles” GROUP BY Month
    The subspace is {City: Los Angeles, Month: *}, the breakdown is {Month} and the measure is {SUM(Sales)}.
    """


    def __init__(self, source_df: pd.DataFrame, subspace: Dict[str, str],
                 breakdown: str | List[str],
                 measure: tuple):
        """
        Initialize the DataScope with the provided subspace, breakdown and measure.

        :param source_df: The DataFrame containing the data.
        :param subspace: dict of filters, e.g., {'City': 'Los Angeles', 'Month': '*'}
        :param breakdown: The dimension(s) to group by. Can be a string or a list of strings.
        :param measure: tuple, (measure_column_name, aggregate_function_name)
        """
        # We want to allow for multi-value groupbys, so we work with lists of strings
        if isinstance(breakdown, str):
            breakdown = [breakdown]
        self.source_df = source_df
        self.subspace = subspace
        self.breakdown = breakdown
        self.measure = measure
        self.breakdown_frozen = frozenset(self.breakdown)
        self.hash = None

    def __hash__(self):
        if self.hash is not None:
            return self.hash
        # Need a hashable representation of subspace for hashing
        subspace_tuple = tuple(sorted(self.subspace.items())) if isinstance(self.subspace, dict) else tuple(
            self.subspace)
        self.hash = hash((subspace_tuple, frozenset(self.breakdown), self.measure))
        return self.hash

    def __repr__(self):
        return f"DataScope(subspace={self.subspace}, breakdown='{self.breakdown}', measure={self.measure})"

    def __eq__(self, other):
        if not isinstance(other, DataScope):
            return False
        return (self.subspace == other.subspace and
                self.breakdown == other.breakdown and
                self.measure == other.measure)

    def apply_subspace(self) -> pd.DataFrame:
        """
        Applies the subspace filters to the source DataFrame and returns the filtered DataFrame.
        """
        filtered_df = self.source_df.copy()
        for dim, value in self.subspace.items():
            if value != '*':
                pattern = rf"^.+<= {dim} <= .+$"
                pattern_matched = re.match(pattern, str(value))
                if pattern_matched:
                    # If the value is a range, split it and filter accordingly
                    split = re.split(r"<=|>=|<|>", value)
                    lower_bound, dim, upper_bound = float(split[0].strip()), split[1].strip(), float(split[2].strip())
                    filtered_df = filtered_df[(filtered_df[dim] >= lower_bound) & (filtered_df[dim] <= upper_bound)]
                else:
                    filtered_df = filtered_df[filtered_df[dim] == value]
        return filtered_df

    def _subspace_extend(self, n_bins: int = 10) -> List['DataScope']:
        """
        Extends the subspace of the DataScope into its sibling group by the dimension dim_to_extend.
        Subspaces with the same sibling group only differ from each other in 1 non-empty filter.

        :param n_bins: The number of bins to use for numeric columns. Defaults to 10.

        :return: A list of new DataScope objects with the extended subspace.
        """
        new_ds = []
        if isinstance(self.subspace, dict):
            for dim_to_extend in self.subspace.keys():
                unique_values = self.source_df[dim_to_extend].dropna().unique()
                # If there are too many unique values, we bin them if it's a numeric column, or only choose the
                # top 10 most frequent values if it's a categorical column
                if len(unique_values) > n_bins:
                    if self.source_df[dim_to_extend].dtype.kind in 'biufcmM':
                        # Bin the numeric column
                        bins = pd.cut(self.source_df[dim_to_extend], bins=n_bins, retbins=True)[1]
                        unique_values = [f"{bins[i]} <= {dim_to_extend} <= {bins[i + 1]}" for i in range(len(bins) - 1)]
                    # else:
                    #     # Choose the top 10 most frequent values
                    #     top_values = self.source_df[dim_to_extend].value_counts().nlargest(10).index.tolist()
                    #     unique_values = [v for v in unique_values if v in top_values]
                for value in unique_values:
                    # Ensure it's a sibling
                    if self.subspace.get(dim_to_extend) != value:
                        # Add the new DataScope with the extended subspace
                        new_subspace = self.subspace.copy()
                        new_subspace[dim_to_extend] = value
                        new_ds.append(DataScope(self.source_df, new_subspace, self.breakdown, self.measure))
        return new_ds

    def _measure_extend(self, measures: List[Tuple[str, str]]) -> List['DataScope']:
        """
        Extends the measure of the DataScope while keeping the same breakdown and subspace.

        :param measures: The measures to extend.
        :return: A list of new DataScope objects with the extended measure.
        """
        new_ds = []
        for measure_col, agg_func in measures:
            if (measure_col, agg_func) != self.measure:
                new_ds.append(DataScope(self.source_df, self.subspace, self.breakdown, (measure_col, agg_func)))
        return new_ds

    def _breakdown_extend(self, dims: List[List[str]]) -> List['DataScope']:
        """
        Extends the breakdown of the DataScope while keeping the same subspace and measure.

        :param dims: The dimensions to extend the breakdown with.
        :return: A list of new DataScope objects with the extended breakdown.
        """
        new_ds = []

        for breakdown_dim in dims:
            if breakdown_dim != self.breakdown:
                new_ds.append(DataScope(self.source_df, self.subspace, breakdown_dim, self.measure))
        return new_ds

    def create_hds(self, dims: List[List[str]] = None,
                   measures: List[Tuple[str,str]] = None, n_bins: int = 10,
                   extend_by_measure: bool = False,
                   extend_by_breakdown: bool = False,
                   ) -> 'HomogenousDataScope':
        """
        Generates a Homogeneous Data Scope (HDS) from a base data scope, using subspace, measure and breakdown
        extensions as defined in the MetaInsight paper.

        :param dims: The temporal dimensions to extend the breakdown with. Expected as a list of strings.
        :param measures: The measures to extend the measure with. Expected to be a dict {measure_column: aggregate_function}.
        :param n_bins: The number of bins to use for numeric columns. Defaults to 10.
        :param extend_by_measure: Whether to use measure extension or not. Defaults to False. Setting this to true
        can lead to metainsights with mixed aggregation functions, which may often be undesirable.
        :param extend_by_breakdown: Whether to use breakdown extension or not. Defaults to False. Setting this to True
        can lead to metainsights with several disjoint indexes, which may often be undesirable.

        :return: A HDS in the form of a list of DataScope objects.
        """
        hds = [self]
        if dims is None:
            dims = []
        if measures is None:
            measures = {}

        # Subspace Extending
        hds.extend(self._subspace_extend(n_bins=n_bins))

        # Measure Extending.
        # We may not want to do it though, if we want our HDS to only contain the original measure.
        if extend_by_measure:
            hds.extend(self._measure_extend(measures))

        # Breakdown Extending
        if extend_by_breakdown:
            hds.extend(self._breakdown_extend(dims))

        return HomogenousDataScope(hds)

    def compute_impact(self) -> float:
        """
        Computes the impact of the data scope based on the provided impact measure.
        We define impact as the proportion of rows between the data scope and the total date scope, multiplied
        by their KL divergence.
        """
        if len(self.subspace) == 0:
            # No subspace, no impact
            return 0
        # Use the provided impact measure or default to the data scope's measure
        impact_col, agg_func = self.measure
        if impact_col not in self.source_df.columns:
            raise ValueError(f"Impact column '{impact_col}' not found in source DataFrame.")

        # Perform subspace filtering
        filtered_df = self.apply_subspace()
        # Group by breakdown dimension and aggregate measure
        if any([True for dim in self.breakdown if dim not in filtered_df.columns]):
            # Cannot group by breakdown if it's not in the filtered data
            return 0
        if impact_col not in filtered_df.columns:
            # Cannot aggregate if measure column is not in the data
            return 0
        try:
            numeric_columns = filtered_df.select_dtypes(include=['number']).columns.tolist()
            # Perform the aggregation
            if agg_func != "std":
                aggregated_series = filtered_df.groupby(impact_col)[numeric_columns].agg(agg_func)
            else:
                # If the aggregation is std, we need to manually provide ddof
                aggregated_series = filtered_df.groupby(impact_col)[numeric_columns].std(ddof=1)
            cache_result = cache.get_from_groupby_cache((impact_col, agg_func))
            if cache_result is not None:
                # If the aggregation is in the cache, use it
                aggregated_source = cache_result
            else:
                if agg_func != "std":
                    aggregated_source = self.source_df.groupby(impact_col)[numeric_columns].agg(agg_func)
                else:
                    # If the aggregation is std, we need to manually provide ddof
                    aggregated_source = self.source_df.groupby(impact_col)[numeric_columns].std(ddof=1)
                # Cache the result of the groupby operation
                cache.add_to_groupby_cache((impact_col, agg_func), aggregated_source)
        except Exception as e:
            # raise e
            print(f"Error during aggregation for {self}: {e}")
            return 0

        kl_divergence = kl_div(aggregated_series, aggregated_source).mean()
        # If it is still a series, then the first mean was on a dataframe and not a series, and thus we need
        # to take the mean to get a float.
        if isinstance(kl_divergence, pd.Series):
            kl_divergence = kl_divergence.mean()
        row_proportion = len(filtered_df.index.to_list()) / len(self.source_df.index.to_list())
        impact = row_proportion * kl_divergence
        return impact

    def create_query_string(self, df_name: str = None) -> str:
        """
        Create a query string for the data scope.
        :param df_name: The name of the DataFrame to use in the query string.
        :return:
        """
        if df_name is None:
            df_name = self.source_df.name if self.source_df.name else "df"
        subspace_where_string = []
        for dim, value in self.subspace.items():
            # If the value is a range, we can just add it as is
            pattern = rf"^.+<= {dim} <= .+$"
            pattern_matched = re.match(pattern, str(value))
            if pattern_matched:
                subspace_where_string.append(value)
            else:
                # Otherwise, we need to add it as an equality string
                subspace_where_string.append(f"{dim} == '{value}'")
        subspace_where_string = 'WHERE ' + ' AND '.join(subspace_where_string)
        measures_select_string = f'SELECT {self.measure[1].upper()}({self.measure[0]})'
        breakdown_groupby_string = f"GROUP BY {self.breakdown}"
        query_string = f"{measures_select_string} FROM {df_name} {subspace_where_string} {breakdown_groupby_string}"
        return query_string






class HomogenousDataScope:
    """
    A homogenous data scope.
    A list of data scopes that are all from the same source_df, and are all created using
    one of the 3 extension methods of the DataScope class.
    """

    def __init__(self, data_scopes: List[DataScope]):
        """
        Initialize the HomogenousDataScope with the provided data scopes.

        :param data_scopes: A list of DataScope objects.
        """
        self.data_scopes = data_scopes
        self.source_df = data_scopes[0].source_df if data_scopes else None
        self.impact = 0

    def __iter__(self):
        """
        Allows iteration over the data scopes.
        """
        return iter(self.data_scopes)

    def __len__(self):
        """
        Returns the number of data scopes.
        """
        return len(self.data_scopes)

    def __getitem__(self, item):
        """
        Allows indexing into the data scopes.
        """
        return self.data_scopes[item]

    def __repr__(self):
        return f"HomogenousDataScope(#DataScopes={len(self.data_scopes)})"

    def __lt__(self, other):
        """
        Less than comparison for sorting.
        :param other: Another HomogenousDataScope object.
        :return: True if this object is less than the other, False otherwise.
        """
        # We use the negative impact, since we want to use a max-heap but only have min-heap available
        return - self.impact < - other.impact

    def compute_impact(self) -> float:
        """
        Computes the impact of the HDS. This is the sum of the impacts of all data scopes in the HDS.
        :return: The total impact of the HDS.
        """
        impact = 0
        for ds in self.data_scopes:
            # Use the cached impact if available to avoid recomputation, since computing the impact
            # is the single most expensive operation in the entire pipeline
            cache_result = cache.get_from_datascope_cache(ds.__hash__())
            if cache_result is not None:
                ds_impact = cache_result
            else:
                ds_impact = ds.compute_impact()
                cache.add_to_datascope_cache(ds.__hash__(), ds_impact)
            impact += ds_impact
        self.impact = impact
        return impact
