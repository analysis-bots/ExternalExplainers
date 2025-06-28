from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series
from pandas.core.interchange.dataframe_protocol import DataFrame
from typing import List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from external_explainers.commons import utils

ALPHA = 0.9
START_BOLD = r'$\bf{'
END_BOLD = '}$'

HIGH = 1
LOW = -1


# BETA = 0.1
def proportion(series):
    c = series.count()
    return (series.count()) / (series.count().sum())


class OutlierExplainer:
    """
    A class for computing the outlier measure and explaining outliers.\n
    This is based on the article "Scorpion: Explaining Away Outliers in Aggregate Queries" by Eugene Wu, Samuel Madden.\n
    The influence of a predicate :math:`p` on an output result :math:`o` is depends on:\n
    .. math:: \\Delta_{agg} (o,p) = agg(g_0) - agg(g_0 - p(g_0))\n
    Where: \n
    - :math:`agg` is the aggregation function.\n
    - :math:`g_0` is a set of input tuples.\n
    In other words, it is the difference between the original result and the result after removing the tuples that satisfy the predicate.\n
    From which we define the influence as:\n
    .. math:: inf_{agg}(o,p) = \\frac{\\Delta_{agg}(o,p)}{\\Delta g_0} = \\frac{\\Delta_{agg}(o,p)}{|p(g_0)|}\n
    We then include the error direction, :math:`d`, to the influence calculation:\n
    .. math:: inf_{agg}(o,p, d) = d \\ * \\ inf_{agg}(o,p) \n
    Next, the user may select a hold-out result, :math:`h`, that the returned predicate should not influence.\n
    Intuitively, :math:`p` should be penalized if it influences the hold-out results in any way.\n
    The influence with the hold-out set is defined as:\n
    .. math:: inf_{agg}(o,p,d,h) = \\lambda \\ * \\ inf_{agg}(o,p,d) - (1 - \\lambda) \\ * \\ inf_{agg}(h,p)\n
    Where:\n
    - :math:`\\lambda` is a parameter that represents the importance of not changing the value of the hold-out set.
    In this project we use :math:`\\lambda = 0.9`.\n
    """

    def __init__(self):
        super().__init__()

    def calc_influence_pred(self, df_before: DataFrame, df_after: DataFrame, target: str, dir: int) -> float:
        """
        Calculate the influence of a predicate on a given target attribute.

        This function computes the influence of a target attribute by comparing its values
        before and after some transformation, adjusted by a direction factor. It also
        considers the influence of other attributes in the DataFrame.

        :param df_before: DataFrame containing the data before the transformation.
        :param df_after: DataFrame containing the data after the transformation.
        :param target (str)) The target attribute for which the influence is being calculated.
        :param dir: Direction factor, either 1 or -1, indicating the direction of the outlier. 1 for high, -1 for low.

        :return: The calculated influence score. Returns -1 if an error occurs during calculation.
        """
        try:
            # Compute target influence - the ratio between the change in the output and the number of
            # tuples that satisfy the predicate, multiplied by the direction factor.
            denominator = df_before[target] + df_after[target]
            # We may have a try catch here, but division by zero is still causing a runtime warning.
            if denominator == 0:
                return -1
            target_inf = ((df_before[target] - df_after[target]) * dir) / denominator
        except:
            return -1

        # Compute the holdout influence - the sum of the square root of the absolute difference between the
        # values of the target attribute for each tuple in the DataFrame before and after the transformation.
        holdout_inf = 0
        for i in df_before.index:
            if i != target:
                try:
                    holdout_inf += np.sqrt(abs(df_after[i] - df_before[i]))
                except:
                    return -1
        # Return the final influence score, calculated as a weighted sum of the target and holdout influences.
        return ALPHA * (target_inf) - (1 - ALPHA) * (holdout_inf / (len(df_before.index)))

    def merge_preds(self, df_agg: DataFrame, df_in: DataFrame, df_in_consider: DataFrame,
                    preds: List[Tuple[str, Tuple[float, float], float, str, int | None]], g_att: str, g_agg: str,
                    agg_method: str, target, dir: int) -> tuple[
        list[tuple[str, tuple[float, float], int | None]], float, Any | None]:
        """
        Merge predicates to find the most influential attributes.

        This function iterates over a list of predicates, applying filters to the input DataFrame to exclude
        certain rows based on the predicates. It calculates the influence of each predicate and keeps track
        of the most influential ones. The final influence score and the filtered DataFrame are returned.

        :param df_agg: DataFrame containing the aggregated data.
        :param df_in: DataFrame containing the input data.
        :param df_in_consider: DataFrame containing the data to be considered for filtering.
        :param preds: A list of tuples containing predicates with attribute name, bin or value, score, kind, and rank.
        :param g_att: The grouping attribute.
        :param g_agg: The aggregation attribute.
        :param agg_method: The aggregation method to be used.
        :param target: The target attribute for which the influence is being calculated.
        :param dir: Direction factor, either 1 or -1, indicating the direction of the outlier. 1 for high, -1 for low.

        :return: A tuple containing a list of final predicates, the final influence score, and the final aggregated DataFrame.
        """
        # Initialize variables to store the final predicates, influence score, and aggregated DataFrame.
        final_pred = []
        final_inf = 0.001
        final_agg_df = None
        final_filter = False
        final_filter_df = False

        prev_attrs = []

        for p in preds:
            attr, i, score, kind, rank = p

            # Avoid going over previously seen attributes
            if attr in prev_attrs:
                continue
            prev_attrs.append(attr)

            # If the kind is 'bin', we use the bin values to filter the DataFrame.
            if kind == 'bin':
                bin = i
                final_filter_test = final_filter | ((df_in_consider[attr] < bin[0]) | (df_in_consider[attr] >= bin[1]))
                final_filter_df_test = final_filter_df | ((df_in[attr] < bin[0]) | (df_in[attr] >= bin[1]))
            # Otherwise, we use the attribute value to filter the DataFrame.
            else:
                final_filter_test = final_filter | (df_in_consider[attr] != i)
                final_filter_df_test = final_filter_df | (df_in[attr] != i)

            # Apply the filter to the input DataFrame and the aggregated DataFrame.
            df_exc_consider = df_in_consider[final_filter_test]
            df_exc_final = df_in[final_filter_df_test]

            # Perform the aggregation operation on the filtered DataFrames.
            agged_val_consider = df_exc_consider.groupby(g_att)[g_agg].agg(agg_method)
            agged_val = df_exc_final.groupby(g_att)[g_agg].agg(agg_method)

            # Normalize the aggregated values if the aggregation method is 'count'.
            if agg_method == 'count':
                agged_val_consider = agged_val_consider / agged_val_consider.sum()
                agged_val = agged_val / agged_val.sum()

            # Compute the influence score for the predicate.
            inf = self.calc_influence_pred(df_agg, agged_val_consider, target, dir) / pow(
                (df_in_consider.shape[0] / (df_exc_consider.shape[0] + 1)), 2)

            # If the influence score is greater by a factor of at least 1.3, update the final predicates.
            if inf / final_inf > 1.3:
                final_pred.append((attr, i, rank))
                final_inf = inf

                final_agg_df = agged_val
                final_filter = final_filter_test
                final_filter_df = final_filter_df_test
            # If the influence score is not much higher, break the loop.
            else:
                break
        return final_pred, final_inf, final_agg_df


    def compute_predicates_per_attribute(self, attr: str, df_in: DataFrame, g_att: str,
                                         g_agg: str, agg_method: str, target: str, dir: int,
                                         df_in_consider: DataFrame, df_agg_consider: DataFrame) -> List[Tuple[str, Any, float, str, int]]:
        """
            Compute predicates for a given attribute.

            This function calculates the influence of various predicates on a target attribute
            by iterating over the values or bins of the given attribute. It generates a list of
            predicates with their corresponding influence scores.

            :param attr: The attribute for which predicates are being computed.
            :param df_in: DataFrame containing the input data.
            :param g_att: The grouping attribute.
            :param g_agg: The aggregation attribute.
            :param agg_method: The aggregation method to be used.
            :param target: The target attribute for which the influence is being calculated.
            :param dir: Direction factor, either 1 or -1, indicating the direction of the outlier.
            :param df_in_consider: DataFrame containing the data to be considered for filtering.
            :param df_agg_consider: DataFrame containing the aggregated data to be considered.

            :return: A list of predicates with their influence scores.
        """
        dtype = df_in[attr].dtype.name
        predicates = []
        exps = {}

        # Ignore attributes with high correlation to the target attribute.
        if dtype in ['int64', 'float64']:
            if (df_in[g_att].dtype.name in ['int64', 'float64'] and df_in[g_att].corr(df_in[attr]) > 0.7) or (
                    df_in[g_agg].dtype.name in ['int64', 'float64'] and df_in[g_agg].corr(df_in[attr]) > 0.7):
                return []

        # Get the series for the attribute and its data type.
        series = df_in[attr]
        dtype = df_in[attr].dtype.name
        flag = False
        df_in_consider_attr = df_in_consider[[g_att, g_agg, attr]]

        # If the data type is not 'float64', calculate the influence score for each value of the attribute.
        if dtype not in ['float64']:
            vals = series.value_counts()
            if dtype != 'int64' or len(vals) < 20:
                # Skip attributes with more than 50 unique values, as they are too computationally expensive.
                if len(vals) > 50:
                    return []
                flag = True

                for i in vals.index:

                    # Exclude rows with the current value of the attribute.
                    df_in_target_exc = df_in_consider_attr[(df_in_consider_attr[attr] != i)]
                    # Aggregate the values for the excluded rows.
                    agged_val = df_in_target_exc.groupby(g_att)[g_agg].agg(agg_method)
                    if agg_method == 'count':
                        agged_val = agged_val / agged_val.sum()

                    # Calculate the influence score for the predicate.
                    inf = self.calc_influence_pred(df_agg_consider, agged_val, target, dir) / (
                        (df_in_consider.shape[0] / (df_in_target_exc.shape[0] + 0.01)))

                    exps[(attr, i)] = inf
                    predicates.append((attr, i, inf, 'cat', None))

        n_bins = 20
        if not flag:
            _, bins = pd.cut(series, n_bins, retbins=True, duplicates='drop')
            df_bins_in = pd.cut(df_in_consider_attr[attr], bins=bins).value_counts(
                normalize=True).sort_index()  # .rename('idx')
            i = 1
            for bin in df_bins_in.keys():
                new_bin = (float("{:.2f}".format(bin.left)), float("{:.2f}".format(bin.right)))
                df_in_exc = df_in_consider_attr[
                    ((df_in_consider_attr[attr] < new_bin[0]) | (df_in_consider_attr[attr] >= new_bin[1]))]
                agged_val = df_in_exc.groupby(g_att)[g_agg].agg(agg_method)
                if agg_method == 'count':
                    agged_val = agged_val / agged_val.sum()

                # Calculate the influence score for the predicate.
                inf = self.calc_influence_pred(df_agg_consider, agged_val, target, dir) / (
                        (df_in_consider_attr.shape[0] / df_in_exc.shape[0]) + 1)

                # Store the influence score for the predicate.
                exps[(attr, (new_bin[0], new_bin[1]))] = inf

                # Add the predicate to the list of predicates.
                predicates.append((attr, new_bin, inf, 'bin', i))
                i += 1

        return predicates


    def pred_to_human_readable(self, non_formatted_pred):
        explanation = f'This outlier is not as significant when excluding rows with:\n'
        for_wizard = ''
        for a, bins in non_formatted_pred.items():
            for b in bins:
                if type(b[0]) is tuple:
                    pred = f"{b[0][0]} < {a} < {b[0][1]}"
                    inter_exp = r'$\bf{{{}}}$'.format(utils.to_valid_latex(pred))
                else:
                    pred = f"{a}={b[0]}"
                    inter_exp = r'$\bf{{{}}}$'.format(utils.to_valid_latex(pred))
                if b[1] is not None:
                    if b[1] <= 5:
                        inter_exp = inter_exp + '-' + r'$\bf{low}$'
                    elif b[1] >= 25:
                        inter_exp = inter_exp + '-' + r'$\bf{high}$'
            inter_exp += '\n'
            for_wizard += inter_exp
            explanation += inter_exp

        return explanation, for_wizard

    def draw_bar_plot(self, df_agg: DataFrame | Series, final_df: DataFrame, g_att: str, g_agg: str, final_pred_by_attr: dict,
                      target: str, agg_title: str, added_text: dict = None) -> None:
        """
        Draw a bar plot to visualize the influence of predicates on the target attribute.

        This function generates a bar plot comparing the aggregated values of the target attribute
        before and after applying the most influential predicates. It highlights the differences
        and provides an explanation for the outlier.

        :param df_agg: DataFrame containing the aggregated data.
        :param final_df: DataFrame containing the final aggregated data after applying predicates.
        :param g_att: The grouping attribute.
        :param g_agg: The aggregation attribute.
        :param final_pred_by_attr: Dictionary containing the final predicates grouped by attribute.
        :param target: The target attribute for which the influence is being visualized.
        :param agg_title: Title for the aggregation method used in the plot.
        :param added_text: Additional text to add to the plot. Optional. Expected: dict with 'text' and 'position' keys.

        :return: None. Displays the bar plot.
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        x1 = list(df_agg.index)
        ind1 = np.arange(len(x1))
        y1 = df_agg.values

        x2 = list(final_df.index)
        ind2 = np.arange(len(x2))
        y2 = final_df.values

        explanation, for_wizard = self.pred_to_human_readable(final_pred_by_attr)

        bar1 = ax.bar(ind1 - 0.2, y1, 0.4, alpha=1., label='All')
        bar2 = ax.bar(ind2 + 0.2, y2, 0.4, alpha=1., label=f'without\n{for_wizard}')
        ax.set_ylabel(f'{g_agg} {agg_title}')
        ax.set_xlabel(f'{g_att}')
        ax.set_xticks(ind1)
        ax.set_xticklabels(tuple([str(i) for i in x1]), rotation=45)
        ax.legend(loc='best')
        ax.set_title(explanation)
        bar1[x1.index(target)].set_edgecolor('tab:green')
        bar1[x1.index(target)].set_linewidth(2)
        bar2[x2.index(target)].set_edgecolor('tab:green')
        bar2[x2.index(target)].set_linewidth(2)
        ax.get_xticklabels()[x1.index(target)].set_color('tab:green')

        plt.tight_layout()

        if added_text is not None:
            # Draw the plot first to establish the bounding boxes.
            plt.draw()
            text = added_text['text']
            position = added_text['position']
            renderer = ax.figure.canvas.get_renderer()
            max_label_height = 0

            for label in ax.get_xticklabels() + [ax.xaxis.get_label()]:
                bbox = label.get_window_extent(renderer=renderer)
                if bbox.height > max_label_height:
                    max_label_height = bbox.height

            if position == "bottom":
                offset_in_points = -(max_label_height + 10)

                ax.annotate(
                    text,
                    xy=(0.5, 0),  # anchor at the bottom of the axes
                    xycoords='axes fraction',
                    xytext=(0, offset_in_points),
                    textcoords='offset points',
                    ha='center', va='top',
                    fontsize=16
                )
            elif position == "top":
                offset_in_points = max_label_height + 10

                ax.annotate(
                    text,
                    xy=(0.5, 1),  # anchor at the top of the axes
                    xycoords='axes fraction',
                    xytext=(0, offset_in_points),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=16
                )

        plt.show()


    def explain(self, df_agg: DataFrame, df_in: DataFrame, g_att: str, g_agg: str, agg_method: str, target: str,
                dir: int, control=None, hold_out: List = None, k: int = 1, draw_plot: bool = True) \
            -> str | None | Tuple:
        """
        Explain the outlier in the given DataFrame.

        This function identifies and explains outliers in the given DataFrame by calculating the influence of
        various attributes on the target attribute. It iterates over the attributes, applies filters, and
        computes the influence score for each attribute. The most influential attributes are then used to
        generate an explanation for the outlier.

        :param df_agg: DataFrame containing the aggregated data.
        :param df_in: DataFrame containing the input data.
        :param g_att: The grouping attribute.
        :param g_agg: The aggregation attribute.
        :param agg_method: The aggregation method to be used.
        :param target: The target attribute for which the influence is being calculated.
        :param dir: Direction factor, either 1 or -1, indicating the direction of the outlier. 1 for high, -1 for low.
        :param control: List of control values for the grouping attribute.
        :param hold_out: List of attributes to be held out from the analysis.
        :param k: Number of top attributes to consider for the explanation.

        :return: None. Will generate a plot with the explanation for the outlier.
        """
        if hold_out is None:
            hold_out = []

        # Get the attributes from the input DataFrame and remove the hold-out attributes.
        attrs = df_in.columns
        attrs = [a for a in attrs if a not in hold_out + [g_att, g_agg]]

        agg_title = agg_method
        if agg_method == 'count':
            df_agg = df_agg / df_agg.sum()

        predicates = []

        # If no control values are provided, use all values for the grouping attribute.
        if control == None:
            control = list(df_agg.index)
        df_in_consider = df_in
        df_agg_consider = df_agg  # [control]

        # Iterate over the attributes, calculate the influence score, and generate predicates.
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.compute_predicates_per_attribute, attr, df_in, g_att, g_agg, agg_method, target,
                                dir, df_in_consider, df_agg_consider) for attr in attrs
            ]
            for future in as_completed(futures):
                preds = future.result()
                predicates += preds


        # Sort the predicates by influence score in descending order.
        predicates.sort(key=lambda x: -x[2])

        # Merge the predicates to find the most influential attributes.
        final_pred, final_inf, final_df = self.merge_preds(df_agg_consider, df_in, df_in_consider, predicates, g_att,
                                                           g_agg, agg_method, target, dir)

        # If the final DataFrame is empty, return an error message. Otherwise, generate the explanation plot.
        if final_df is None:
            return "There was no explanation."

        # Create a new DataFrame with the aggregated values and the control values.
        new_df_agg = df_agg.copy()
        new_df_agg[control] = final_df[control]
        new_df_agg[target] = final_df[target]
        final_pred_by_attr = {}

        # Group the final predicates by attribute.
        for a, i, rank in final_pred:
            if a not in final_pred_by_attr.keys():
                final_pred_by_attr[a] = []
            final_pred_by_attr[a].append((i, rank))

        # Create a plot to display the explanation for the outlier, or return everything needed to draw the plot later.
        if draw_plot:
            self.draw_bar_plot(df_agg, final_df, g_att, g_agg, final_pred_by_attr, target, agg_title)
            return None
        else:
            return df_agg, final_df, g_att, g_agg, final_pred_by_attr, target, agg_title
