import pandas as pd
from itertools import combinations



def load_dataset(path):
    """ Load the dataset """
    
    if ".csv" in path: dataset = pd.read_csv(path)
    else: dataset = pd.read_excel(path)
    if "Unnamed: 0" in dataset.columns: dataset = dataset.drop("Unnamed: 0", axis=1)
    return dataset


def save_dataset(dataset, path, mode="w", index=False, header=True):
    """ Save the dataset """
    
    if ".csv" in path: dataset.to_csv(path, mode=mode, index=index, header=header)
    else: dataset.to_excel(path)


def balance_dataset(dataset_train, label):
    """ Undersample the majority class to match the 
    number of samples in the minority class.
    Possible classes are 0 or 1 """

    class_0 = dataset_train[dataset_train[label] == 0]
    class_1 = dataset_train[dataset_train[label] == 1]

    if len(class_0) == len(class_1):
        print("Dataset balanced yet!\nNo balancing needed\n")
        return dataset_train

    if len(class_1) < len(class_0):
        class_undersampled = class_0.sample(n=len(class_1), replace=False, random_state=42)
        balancedDataset = pd.concat([class_undersampled, class_1])
    else:
        class_undersampled = class_1.sample(n=len(class_0), replace=False, random_state=42)
        balancedDataset = pd.concat([class_undersampled, class_0])

    # Shuffle dataset
    return balancedDataset.sample(frac=1, random_state=42).reset_index(drop=True)


def create_pairs(dataset) -> pd.DataFrame:
    """ Create pairs of names with their labels. 
        We use the symbol '@' to separate the names.
    """
    
    pairs = []
    labels = []
    grouped = dataset.groupby('AccountNumber')
    
    if "cluster" in dataset.columns:
        dataset.fillna(0, inplace=True)
        
        for _, group in grouped:
            names = group['Name'].tolist()
            clusters = group['cluster'].tolist()
            if(len(names)) == 1:
                pairs.append(" @ ".join([names[0], names[0]]))
                labels.append(0)
            else:
                for (name1, cluster1), (name2, cluster2) in combinations(zip(names, clusters), 2):
                    pairs.append(" @ ".join([name1, name2]))
                    labels.append(0 if cluster1 == cluster2 else 1)
            
    else:
        for _, group in grouped:
            names = group['Name'].tolist()
            holders = group['Holder'].tolist()
            if(len(names)) == 1:
                pairs.append(" @ ".join([names[0], names[0]]))
                labels.append(0)
            else:
                for (name1, holder1), (name2, holder2) in combinations(zip(names, holders), 2):
                    pairs.append(" @ ".join([name1, name2]))
                    labels.append(0 if holder1 == holder2 else 1)
    
    
    df = pd.DataFrame()
    df['text'] = pairs
    df['label'] = labels
    return df


def prepocess_dataset(dataset):
    """ Simple remove the unused columns and remove all the duplicates """
    
    if "Address" in dataset.columns:
        preprocessDataset = dataset.drop("Address", axis=1)
        return preprocessDataset.drop_duplicates()
    return dataset.drop_duplicates()


def tokenize_dataset(dataframe, tokenizer) -> list:
    """ 
        Tokenize the dataset for the encoding layer of the CharacterBERT model.
        The tokenization is done with the symbol '@' to separate the names.
    """
    # return dataframe['text'].apply(lambda x: ['[CLS]', *[y.strip() for y in x.split("@")], '[SEP]'])
    if not tokenizer:
        return dataframe['text'].apply(lambda x: ['[CLS]'] + [y.strip() for y in x.split("@")[0].split()] + ['[SEP]'] + [y.strip() for y in x.split("@")[1].split()] + ['[SEP]'])
    else:
        return dataframe['text'].apply(lambda x: ['[CLS]'] + tokenizer.tokenize(' '.join([x.split("@")[0]])) + ['[SEP]'] + tokenizer.tokenize(' '.join([x.split("@")[1]])) + ['[SEP]'])
        

def tokenize_dataset_pair(dataframe, tokenizer) -> list:
    if not tokenizer:
        return dataframe['text'].apply(
            lambda x: 
                (['[CLS]'] + [y.strip() for y in x.split("@")[0].split()] + ['[SEP]'],
                ['[CLS]'] + [y.strip() for y in x.split("@")[1].split()] + ['[SEP]'])
            )
    else:
        return dataframe['text'].apply(
            lambda x: 
                (['[CLS]'] + tokenizer.tokenize(' '.join([x.split("@")[0]])) + ['[SEP]'],
                ['[CLS]'] + tokenizer.tokenize(' '.join([x.split("@")[1]])) + ['[SEP]'])
            )
