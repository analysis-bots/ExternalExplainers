from external_explainers import OutlierExplainer
from tests.utils import get_dataset

import os
import json
from colorama import Fore
from colorama import init as colorama_init
import argparse

# Allow the user to save new results to a file, instead of running the test
parser = argparse.ArgumentParser(description='Reproduce the outlier explainer results.')
parser.add_argument('--save', action='store_true', help='Save the reproduced results to a file, overwriting the existing file. Only use this if you are certain that the reproduced results are correct.')

colorama_init(autoreset=True)

def run_outlier_explainer(dataset, gb_on, agg_function, select, dir, target) -> dict:
    """
    Run the outlier explainer on the dataset, saving the intermediate results to a dictionary.
    :param dataset: The dataset to run the outlier explainer on.
    :param gb_on: The attributes to group by.
    :param agg_function: The aggregation function.
    :param select: Which column to select after grouping.
    :param dir: The direction of the outlier.
    :param target: The target value for the outlier explanation.
    :return: A dict containing the intermediate results of the outlier explainer. That is, the computed predicates and the final predictions.
    """
    gb_result = dataset.groupby(gb_on)[select].agg(agg_function)

    results_dict = {}

    measure = OutlierExplainer()

    preds = []

    attrs = dataset.columns
    attrs = [a for a in attrs if a not in [gb_on, select]]

    # Do the predicate calculation for each attribute, like in the explain function.
    for attr in attrs:
        predicates = measure.compute_predicates_per_attribute(
            attr=attr,
            df_in=dataset,
            g_att=gb_on,
            g_agg=select,
            agg_method=agg_function,
            target=target,
            dir=dir,
            df_in_consider=dataset,
            df_agg_consider=gb_result
        )

        preds += predicates

    preds.sort(key=lambda x: -x[2])

    results_dict['preds'] = preds

    # Use the merge_preds function to get the final results.
    final_pred, final_inf, final_df = measure.merge_preds(
        df_agg=gb_result,
        df_in=dataset,
        df_in_consider=dataset,
        preds=preds,
        g_att=gb_on,
        g_agg=select,
        agg_method=agg_function,
        target=target,
        dir=dir
    )

    # Save, while making sure that the index also gets saved.
    if final_df is not None:
        final_df = final_df.to_frame()
        final_df = final_df.to_json(orient='columns')

    results_dict['final_pred'] = final_pred
    results_dict['final_inf'] = final_inf
    results_dict['final_df'] = final_df

    results_dict['gb_on'] = gb_on
    results_dict['agg_function'] = agg_function
    results_dict['select'] = select
    results_dict['dir'] = dir
    results_dict['target'] = target

    return results_dict


def main():

    args = parser.parse_args()

    with open(r"resources/results/outlier_explainer_results.json", "r") as read_file:
        saved_results = json.load(read_file)

    reproduced_results = {}
    # For each dataset in the saved_results, load the dataset, then re-produce the results
    for dataset_name in saved_results.keys():
        dataset = get_dataset(dataset_name)
        values = saved_results[dataset_name]
        # Run the outlier explainer on the dataset
        reproduced_results[dataset_name] = run_outlier_explainer(
            dataset=dataset,
            gb_on=values['gb_on'],
            agg_function=values['agg_function'],
            select=values['select'],
            dir=values['dir'],
            target=values['target']
        )

    if args.save:
        with open(r"resources/results/outlier_explainer_results.json", "w") as write_file:
            json.dump(reproduced_results, write_file, indent=4)
        print(f"{Fore.GREEN}Reproduced results saved to file.")
        exit(0)

    # Temporarily save the reproduced results to a json file
    with open(r"resources/results/reproduced_outlier_explainer_results.json", "w") as write_file:
        json.dump(reproduced_results, write_file, indent=4)

    # Load the reproduced results and delete the file. We do this because saving the original results to file means we may lose some precision
    # in the floating points, and so, instead of dealing with that, we just save the reproduced results to a file and then load it back.
    with open(r"resources/results/reproduced_outlier_explainer_results.json", "r") as read_file:
        reproduced_results = json.load(read_file)

    os.remove(r"resources/results/reproduced_outlier_explainer_results.json")

    total_errors = 0
    errors_on_datasets = []
    ok_datasets = []
    # Compare the saved results with the reproduced results
    for dataset_name in saved_results.keys():
        errors = []
        # Compare the 'preds' list:
        saved_preds = saved_results[dataset_name]['preds']
        reproduced_preds = reproduced_results[dataset_name]['preds']
        if len(saved_preds) != len(reproduced_preds):
            errors.append(f"{Fore.RED} Error in preds length. Expected {len(saved_preds)}, got {len(reproduced_preds)}. These are both sorted lists that should have the same length.")
        for i in range(len(saved_preds)):
            if saved_preds[i] != reproduced_preds[i]:
                errors.append(f"{Fore.RED} Error in preds at index {i}. Expected {saved_preds[i]}, got {reproduced_preds[i]}")
        # Compare the 'final_pred' value:
        saved_final_pred = saved_results[dataset_name]['final_pred']
        reproduced_final_pred = reproduced_results[dataset_name]['final_pred']
        if saved_final_pred != reproduced_final_pred:
            errors.append(f"{Fore.RED} Error in final_pred. Expected {saved_final_pred}, got {reproduced_final_pred}")
        # Compare the 'final_inf' value:
        saved_final_inf = saved_results[dataset_name]['final_inf']
        reproduced_final_inf = reproduced_results[dataset_name]['final_inf']
        if saved_final_inf != reproduced_final_inf:
            errors.append(f"{Fore.RED} Error in final_inf. Expected {saved_final_inf}, got {reproduced_final_inf}")
        # Compare the 'final_df' value:
        saved_final_df = saved_results[dataset_name]['final_df']
        reproduced_final_df = reproduced_results[dataset_name]['final_df']
        if saved_final_df != reproduced_final_df:
            errors.append(f"{Fore.RED} Error in final_df. Expected {saved_final_df}, got {reproduced_final_df}")

        if len(errors) > 0:
            print(f"{Fore.RED} ----------------------------------------------------------------------")
            print(f"{Fore.RED} Errors in dataset '{dataset_name}'. {len(errors)} errors found:")
            for error in errors:
                print(f"\t \t {error}")
            total_errors += len(errors)
            errors_on_datasets.append(dataset_name)
            print(f"{Fore.RED} ----------------------------------------------------------------------\n")
        else:
            print(f"{Fore.GREEN} ----------------------------------------------------------------------")
            print(f"{Fore.GREEN} No errors in dataset '{dataset_name}'.")
            print(f"{Fore.GREEN} ----------------------------------------------------------------------\n")
            ok_datasets.append(dataset_name)

    print(f"{Fore.CYAN} ----------------------------------------------------------------------")
    print(f"{Fore.CYAN} Summary:")
    if total_errors == 0:
        print(f"\t{Fore.GREEN} All tests passed.")
    else:
        print(f"\t{Fore.RED} {total_errors} errors found.")
        print(f"\t{Fore.RED} Errors found in the following datasets:")
        for dataset_name in errors_on_datasets:
            print(f"\t \t{Fore.RED}{dataset_name}")
    if len(ok_datasets) > 0:
        print(f"\t{Fore.GREEN} No errors found in the following datasets:")
        for dataset_name in ok_datasets:
            print(f"\t \t{Fore.GREEN}{dataset_name}")
    print(f"{Fore.CYAN} ----------------------------------------------------------------------")




if __name__ == '__main__':
    main()