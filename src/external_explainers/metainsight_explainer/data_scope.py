import pandas as pd
from typing import Dict, List, Tuple
from scipy.special import kl_div
from concurrent.futures import ThreadPoolExecutor
import time


class DataScope:
    """
    A data scope, as defined in the MetaInsight paper.
    Contains 3 elements: subspace, breakdown and measure.
    Example: for the query SELECT Month, SUM(Sales) FROM DATASET WHERE City==“Los Angeles” GROUP BY Month
    The subspace is {City: Los Angeles, Month: *}, the breakdown is {Month} and the measure is {SUM(Sales)}.
    """

    def __init__(self, source_df: pd.DataFrame, subspace: Dict[str, str], breakdown: str, measure: tuple):
        """
        Initialize the DataScope with the provided subspace, breakdown and measure.

        :param source_df: The DataFrame containing the data.
        :param subspace: dict of filters, e.g., {'City': 'Los Angeles', 'Month': '*'}
        :param breakdown: str, the dimension for group-by
        :param measure: tuple, (measure_column_name, aggregate_function_name)
        """
        self.source_df = source_df
        self.subspace = subspace
        self.breakdown = breakdown
        self.measure = measure

    def __hash__(self):
        # Need a hashable representation of subspace for hashing
        subspace_tuple = tuple(sorted(self.subspace.items())) if isinstance(self.subspace, dict) else tuple(
            self.subspace)
        return hash((subspace_tuple, self.breakdown, self.measure))

    def __repr__(self):
        return f"DataScope(subspace={self.subspace}, breakdown='{self.breakdown}', measure={self.measure})"

    def _subspace_extend(self) -> List['DataScope']:
        """
        Extends the subspace of the DataScope into its sibling group by the dimension dim_to_extend.
        Subspaces with the same sibling group only differ from each other in 1 non-empty filter.

        :return: A list of new DataScope objects with the extended subspace.
        """
        new_ds = []
        if isinstance(self.subspace, dict):
            for dim_to_extend in self.subspace.keys():
                unique_values = self.source_df[dim_to_extend].dropna().unique()
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

    def _breakdown_extend(self, temporal_dimensions: List[str]) -> List['DataScope']:
        """
        Extends the breakdown of the DataScope while keeping the same subspace and measure.

        :param temporal_dimensions: The temporal dimensions to extend the breakdown with.
        :return: A list of new DataScope objects with the extended breakdown.
        """
        new_ds = []

        temporal_dimensions = [d for d in temporal_dimensions if
                               self.source_df[d].dtype in ['datetime64[ns]', 'period[M]', 'int64']]
        for breakdown_dim in temporal_dimensions:
            if breakdown_dim != self.breakdown:
                new_ds.append(DataScope(self.source_df, self.subspace, breakdown_dim, self.measure))
        return new_ds

    def create_hds(self, temporal_dimensions: List[str] = None,
                   measures: List[Tuple[str,str]] = None) -> 'HomogenousDataScope':
        """
        Generates a Homogeneous Data Scope (HDS) from a base data scope, using subspace, measure and breakdown
        extensions as defined in the MetaInsight paper.

        :param temporal_dimensions: The temporal dimensions to extend the breakdown with. Expected as a list of strings.
        :param measures: The measures to extend the measure with. Expected to be a dict {measure_column: aggregate_function}.

        :return: A HDS in the form of a list of DataScope objects.
        """
        hds = [self]
        if temporal_dimensions is None:
            temporal_dimensions = []
        if measures is None:
            measures = {}

        # Subspace Extending
        hds.extend(self._subspace_extend())

        # Measure Extending
        hds.extend(self._measure_extend(measures))

        # Breakdown Extending
        hds.extend(self._breakdown_extend(temporal_dimensions))

        return HomogenousDataScope(hds)

    def compute_impact(self, precomputed_source_df: pd.DataFrame = None) -> float:
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
        filtered_df = self.source_df.copy()
        for dim, value in self.subspace.items():
            if value != '*':
                filtered_df = filtered_df[filtered_df[dim] == value]
        # Group by breakdown dimension and aggregate measure
        if self.breakdown not in filtered_df.columns:
            # Cannot group by breakdown if it's not in the filtered data
            return 0
        if impact_col not in filtered_df.columns:
            # Cannot aggregate if measure column is not in the data
            return 0
        try:
            numeric_columns = filtered_df.select_dtypes(include=['number']).columns.tolist()
            # Perform the aggregation
            aggregated_series = filtered_df.groupby(impact_col)[numeric_columns].agg(agg_func)
            if precomputed_source_df is None:
                aggregated_source = self.source_df.groupby(impact_col)[numeric_columns].agg(agg_func)
            else:
                aggregated_source = precomputed_source_df.groupby(impact_col)[[numeric_columns]].agg(agg_func)
        except Exception as e:
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
        # with ThreadPoolExecutor() as executor:
        #     # Compute the impact of each data scope in parallel
        #     futures = [executor.submit(ds.compute_impact) for ds in self.data_scopes]
        #     for future in futures:
        #         impact += future.result()
        for ds in self.data_scopes:
            impact += ds.compute_impact()
        self.impact = impact
        return impact
