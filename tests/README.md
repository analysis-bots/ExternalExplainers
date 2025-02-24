# Test for the ExternalExplainers package

## Outlier explainer tests
The outlier explainer has the outlier_explainer_tests.py file.\
This file runs the outlier in a predefined manner on some predefined datasets, and checks if the results are as expected.\
The definitions of the datasets and what to run are in the `resources/results/outlier_explainer_results.json` file,
which also contains the expected results.\
\
To run the tests:
```bash
python outlier_explainer_tests.py
```
This will print a report of the tests, and if any of them failed, it will print the expected and actual results.\
\
If you would like to replace the result file instead of running the tests, run the script with the `--save` option:
```bash
python outlier_explainer_tests.py --save
```
This will run the tests and overwrite the result file with the new results.\
Only use this if you are certain that the new results are correct, and that there was some change in the expected results.

## Utils
The `download_datasets_util.py` file will automatically download the 4 datasets used in the tests so far.\
If you would like to add datasets, add more to the `get_dataset` function in `utils.py`.