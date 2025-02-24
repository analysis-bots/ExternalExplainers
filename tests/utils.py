import pandas as pd

def get_dataset(dataset_name: str):
    """
    Loads a dataset from the resources/datasets folder, or downloads it from the web if it does not exist.
    :param dataset_name:
    :return: The dataset as a pandas DataFrame, possibly with some pre-defined columns already selected.
    """
    # If the resources/datasets folder does not exist, create it
    import os
    if not os.path.exists('resources/datasets'):
        os.makedirs('resources/datasets')

    if dataset_name == 'adult':
        try:
            return pd.read_csv('resources/datasets/adult.csv')
        except FileNotFoundError:
            dataset = pd.read_csv('https://raw.githubusercontent.com/analysis-bots/pd-explain/refs/heads/main/Examples/Datasets/adult.csv')
            dataset.to_csv('resources/datasets/adult.csv', index=False)
            return dataset
    elif dataset_name == 'spotify':
        try:
            return pd.read_csv('resources/datasets/spotify_all.csv')
        except FileNotFoundError:
            dataset = pd.read_csv('https://raw.githubusercontent.com/analysis-bots/pd-explain/refs/heads/main/Examples/Datasets/spotify_all.csv')
            dataset.to_csv('resources/datasets/spotify_all.csv', index=False)
            return dataset
    elif dataset_name == 'bank_churners':
        try:
            dataset =  pd.read_csv('resources/datasets/bank_churners.csv')
        except FileNotFoundError:
            dataset = pd.read_csv('https://raw.githubusercontent.com/analysis-bots/pd-explain/refs/heads/main/Examples/Datasets/bank_churners_user_study.csv')
            dataset.to_csv('resources/datasets/bank_churners.csv', index=False)
        return dataset[["Customer_Age", "Dependent_count", "Months_on_book", "Months_Inactive_Count_Last_Year", "Contacts_Count_Last_Year", "Credit_Limit", "Credit_Used", "Credit_Open_To_Buy", "Total_Amount_Change_Q4_vs_Q1", "Total_Transitions_Amount", "Total_Count_Change_Q4_vs_Q1", "Credit_Avg_Utilization_Ratio"]]
    elif dataset_name == 'houses':
        try:
            dataset =  pd.read_csv('resources/datasets/houses.csv')
        except FileNotFoundError:
            dataset = pd.read_csv('https://raw.githubusercontent.com/analysis-bots/pd-explain/refs/heads/main/Examples/Datasets/houses.csv')
            dataset.to_csv('resources/datasets/houses.csv', index=False)
        return dataset[["MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "1stFlrSF", "2ndFlrSF", "GrLivArea", "FullBath", "TotRmsAbvGrd", "GarageCars", "PoolArea", "YrSold", "SalePrice"]]
