import pandas as pd
import json
import time
from lib.plot import *
from lib.saveOutput import *
from lib.datasetManipulation import *
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from lib.download import download_pre_trained_model
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
from lib.trainingUtilities import EarlyStopping

download_pre_trained_model()
from lib.CBertClassif import *

# Load Custom model
model = CBertClassif()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


# New log File
with open('./config/parameters.json', "r") as data_file:
    parameters = json.load(data_file)


# Retrieve parameters
# train_proportion = parameters['train_proportion']
# val_proportion = parameters['val_proportion']
# test_proportion = parameters['test_proportion']
# num_epochs = parameters['num_epochs']
batch_size = parameters['batch_size']
weight_decay = parameters['weight_decay']
learning_rate = parameters['learning_rate']



def create_pairs_for_clustering(dataset):
    """Create pairs of names with their labels"""
    
    pairs = []
    labels = []
    ibans = []
    names1 = []
    names2 = []
    isShared = []
    grouped = dataset.groupby('AccountNumber')

    for iban, group in grouped:
        names = group['Name'].tolist()
        holders = group['Holder'].tolist()
        shared = group['IsShared'].iloc[0]
        
        if(len(names)) == 1:
            #print(iban)
            pairs.append(" @ ".join([names[0], names[0]]))
            labels.append(0)
            ibans.append(iban)
            isShared.append(shared)
            names1.append(names[0])
            names2.append(names[0])
        else:
            for (name1, holder1), (name2, holder2) in combinations(zip(names, holders), 2):
                if(isinstance(name1, float) or isinstance(name2, float)):
                    print("BOOOO FLOAT")
                    print(iban)
                    print(name1, name2)
                pairs.append(" @ ".join([name1, name2]))
                labels.append(0 if holder1 == holder2 else 1)
                ibans.append(iban)
                isShared.append(shared)
                names1.append(name1)
                names2.append(name2)     
        
    
    df = pd.DataFrame()
    df['iban'] = ibans
    df['text'] = pairs
    df['name1'] = names1
    df['name2'] = names2
    df['label'] = labels
    df['IsShared'] = isShared
    
    return df


def fine_tuning(dataset, balance=False, num_epochs=5, train_proportion=0.9, test_proportion=0.1):
    
    """ 
    The dataset is a .csv or .xlsx file. From an existing dataset the program 
    generate another dataset with the couples Name1 - Name2 - Label. 
    On this new dataset the model is trained to recognize the couples.
    """

    if (train_proportion + test_proportion) != 1.0:
        raise Exception("The train_proportion + val_proportion + test_proportion must be equal to 1.0")
        exit()
    

    # -------------------------------------------------------
    # Preprocessing dataset. The preprocessing remove the "address" column
    # Eventually - ballancing the dataset on the IsShared column
    # -------------------------------------------------------
    
    dataset = prepocess_dataset(dataset)
    if balance:
        dataset = balance_dataset(dataset,"IsShared")
        print("\nBalancing the dataset on the isShared column")
        print("Dataset, isShared statistics:")
        print(str(dataset.groupby('IsShared').size()))
    else:
        print("\nDataset, isShared statistics")
        print(str(dataset.groupby('IsShared').size()))
            
    
    
    # -------------------------------------------------------
    # Create the dataset of pairs for the couple prediction task
    # Eventually - ballancing the dataset on the label column
    # -------------------------------------------------------
    
    dataframe = create_pairs(dataset)
    # print("Preview of the dataset for the couple prediction taks:\n")
    # print(dataframe.head(10).to_markdown())
    
    if balance:
        dataframe = balance_dataset(dataframe, "label")
        print("\n\nBalancing the new dataset on the labels column")
        print("Dataset, label statistics")
        print(str(dataframe.groupby('label').size()))
    else:
        print("\nDataset, label statistics")
        print(str(dataframe.groupby('label').size()))
    
    
    
    # -------------------------------------------------------
    # Splitting dataset
    # -------------------------------------------------------
    tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')
    X = tokenize_dataset(dataframe, tokenizer).tolist()
    y = dataframe['label'].tolist()
    # print("\n- Preview of the dataset after the tokenization step:")
    # for i in range(5):print(str(X[i]))
    # print("")
    
    
    if train_proportion >= 1.0:
        X_train = X
        y_train = y
    else: 
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, random_state=42, stratify=y)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, random_state=42, stratify=y_test)

        # # Print the class distribution
        # print("Original data: " + str(Counter(y)))
        # print("Training data: " + str(Counter(y_train)))
        # print("Test data: " + str(Counter(y_test)))
        # print("Validation data: " + str(Counter(y_val)))
        # print("")

        # print("Training set size: " + str(len(X_train)))
        # print("Validation set size: " + str(len(X_val)))
        # print("Test set size: " + str(len(X_test)))
        # print("")

    # Free memory
    del dataset
    del dataframe



    # -------------------------------------------------------
    # Train the model
    # -------------------------------------------------------

    # used parameters until: 5/11 ---> optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=0.0005)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=(len(X_train) // batch_size)*num_epochs)
    criterion = nn.BCELoss()

    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    validation_f1 = []

    early_stopping_train = EarlyStopping(patience=3, delta=0.001)
    early_stopping_val = EarlyStopping(patience=3, delta=0.001)
    # print("\n\nTraining step")
    init = time.time()
    
    
    for epoch in range(num_epochs):
       
        # --------------------------------------
        # Training step
        # --------------------------------------
        print("\n\n------ Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + " ------\n")
        loss_t, accuracy_t = train(model, X_train, y_train, batch_size, optimizer, criterion, scheduler)
        training_loss.append(loss_t)
        training_accuracy.append(accuracy_t)
        print(f'- Training set loss: {loss_t}')
        print(f'- Training set accuracy: {accuracy_t} \n')
        
        # --------------------------------------
        # Evaluate on validation set
        # --------------------------------------
        if train_proportion < 1.0:
            loss_v, metrics, _, _ = test(model, X_val, y_val, batch_size, criterion)
            validation_loss.append(loss_v)
            validation_accuracy.append(metrics["accuracy"])
            validation_f1.append(metrics['f1'])
            # saveCriteria(model, (epoch + 1), loss_t, loss_v)
            
            print(f'- Validation set loss: {loss_v}')
            for el in metrics: print("- " + el +  ":" + str(metrics[el]))
            print(" ")
        
        
            # Evaluate The early stopping
            if early_stopping_train(loss_t) or early_stopping_val(loss_v):
                print("\nEarly stopping after {} epochs!".format(epoch + 1))
                break
        else:
            
            if early_stopping_train(loss_t):
                print("\nEarly stopping after {} epochs!".format(epoch + 1))
                break
        
        torch.cuda.empty_cache()
        
    end = time.time()
    # print("\n\nTraining time: " + str((end - init) / 60) + " minutes")
    

    # Evaluate on test set
    if test_proportion > 0:
        print("\nTesting on Test Set\n")
        _, test_metrics, _, _ = test(model, X_test, y_test, batch_size, criterion)
        for el in test_metrics: print("- " + el +  ":" + str(test_metrics[el]))
        print()
    
    metrics = {}
    metrics["training_time"] = (end - init) / 60
    metrics["training_loss"] = training_loss[-1]
    metrics["training_accuracy"] = training_accuracy[-1]
    
    if test_proportion > 0:
        metrics["test_accuracy"] = test_metrics['accuracy']
        metrics["test_precision"] = test_metrics['precision']
        metrics["test_recall"] = test_metrics['recall']
        metrics["test_f1"] = test_metrics['f1']
          
    return metrics, model
    

def clustering(model, dataset):
    """ """
    
    DATE_NAME = str(datetime.now()).split(".")[0].replace(" ", "_") 
    DATASET_BUILD = "./cross_validation_results/labelled_dataset/labelled_testSet_" + DATE_NAME + ".csv"
    JSON_NAME = "./cross_validation_results/clusters/Clusters_" + DATE_NAME + ".json"
    final_metrics = {}
    
    # preprocess dataset
    dataset = prepocess_dataset(dataset)
    final_metrics['dataset_length'] = len(dataset)
    
    # create pairs
    dataframe = create_pairs_for_clustering(dataset)
    
    # tokenize pairs
    tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')
    X = tokenize_dataset(dataframe, tokenizer).tolist()
    y = dataframe['label'].tolist()
    dataframe = dataframe.drop('text', axis=1)
    
    # Evaluate model
    criterion = nn.BCELoss()
    _, metrics, predictions, total_labels = test(model, X, y, batch_size, criterion)
    
    # print results
    dataframe['predicted'] = predictions
    
    # Free memory
    del total_labels
    del X
    del y
    
    
    account_entities = {}
    dataframe_groupped = dataframe.groupby("iban")
    final_metrics['iban_number'] = len(dataframe_groupped)
    final_metrics['couple_prediction_task'] = metrics
    
    for _, group in dataframe_groupped:
        shared = group['IsShared'].iloc[0]
        iban = group['iban'].iloc[0]
        names1 = group['name1'].tolist()
        names2 = group['name2'].tolist()
        predicted = group['predicted'].tolist()
        account_entities[iban] = {
            'IsShared': int(shared),
            'predicted_shared': -1,
            'real_holders': list(set(dataset[dataset['AccountNumber'] == iban]['Holder'].tolist())),
            'holders': []
        }
        
        

        G = nx.Graph()
        G.add_nodes_from(list(set(names1) | set(names2)))
        for i in range(len(predicted)): # Add edges based on predictions
            if predicted[i] == 0: G.add_edge(names1[i], names2[i])

        if len(predicted) != len(names1):
            print("CHEEEEEEEEEEEEEEEEEEEEEEECK !!!!!!!!!!!!!!")
            print(iban +  " " + str(len(predicted)) + " " + str(len(names1)) + " " + str(len(names2)))


        clusters = list(nx.connected_components(G))
        for cluster in clusters:
            cluster_list = list(cluster)
            representative_name = max(cluster, key=len)
            r_nodes = [el for el in cluster_list]
            account_entities[iban]['holders'].append({
                    "cluster_name": representative_name,
                    "names_list": r_nodes,
                    "holder_from_cluster_name": dataset[(dataset['Name'] == representative_name) & (dataset['AccountNumber'] == iban)]['Holder'].tolist()[0]
            })
        
    

    #### Is shared prediction
    for a in account_entities:
        if len(account_entities[a]['holders']) > 1: account_entities[a]['predicted_shared'] = 1
        elif len(account_entities[a]['holders']) == 1: account_entities[a]['predicted_shared'] = 0
            
    
    #### CORRECTION ON PREDICTION HOLDERS
    for a in account_entities:
        if account_entities[a]['predicted_shared'] == 1 and account_entities[a]['IsShared'] == 0:
            for i,holder in enumerate(account_entities[a]['holders']):
                holder['holder_from_cluster_name'] = holder['holder_from_cluster_name'] + "_" + str(i)
                
    
    
    #### ISSHARED ACCURACY
    predictions = [account_entities[el]['predicted_shared'] for el in account_entities]
    real = [account_entities[el]['IsShared'] for el in account_entities]
    count = 0
    for a in account_entities:
        if account_entities[a]['IsShared'] == account_entities[a]['predicted_shared']:
            count += 1
        
    
    #### TRANSACTION HOLDER PREDICTION
    dataset['Predicted_Holder'] = ["" for el in range(len(dataset))]
    dataset['Representative_name'] = ["" for el in range(len(dataset))]
    for iban in account_entities:
        holder_dict = {}
        representative_names = {}
        for holder in account_entities[iban]['holders']:
            for name in holder["names_list"]:       
                holder_dict[name] = holder['holder_from_cluster_name']
                representative_names[name] = holder['cluster_name']
        
        for index, row in dataset.loc[dataset['AccountNumber'] == iban].iterrows():
            dataset.loc[index,"Predicted_Holder"] = holder_dict[row['Name']]
            dataset.loc[index,"Representative_name"] = representative_names[row['Name']]
            
    number_transaction_ok = len(dataset.loc[dataset['Holder'] == dataset['Predicted_Holder']])
    
    
    
    #### CLUSTERED IBAN PREDICTION
    final_metrics['wrong_iban'] = []
    number_cluster_iban_ok = 0
    shared_not_clustered_iban = 0
    for iban, group in dataset.groupby('AccountNumber'):
        predicted_holder = group['Predicted_Holder'].tolist()
        holder = group['Holder'].tolist()
        check = [predicted_holder[i] == holder[i] for i in range(len(predicted_holder))]
        number_cluster_iban_ok += 1 if all(check) else 0
        if not all(check): 
            final_metrics['wrong_iban'].append(iban)
            # print("IBAN: " + iban + " not correctly clustered! --> " + "Transaction OK: " + str(len([el for el in check if el == True])) + " / " + str(len(check)))
            if dataset.loc[dataset['AccountNumber'] == iban]['IsShared'].tolist()[0] == 1: shared_not_clustered_iban += 1
        
    
    
    # IS_SHARED
    metrics = compute_metrics(predictions, real)
    final_metrics['isShared_task'] = metrics
    final_metrics['correct_clustered_task'] = {
        "accuracy": number_cluster_iban_ok / len(dataframe_groupped),
    }
    if len(dataframe_groupped) - number_cluster_iban_ok > 0:
        final_metrics['correct_clustered_task']['shared_wrong_clustered'] = str(shared_not_clustered_iban)
        final_metrics['correct_clustered_task']['not_shared_wrong_clustered'] = str(len(dataframe_groupped) - number_cluster_iban_ok - shared_not_clustered_iban)
    

    final_metrics['transaction_holder_task'] = {
        "accuracy": number_transaction_ok / len(dataset)
    }
    
    #### Export labelled dataset
    dataset.to_csv(DATASET_BUILD, index=False)
    json.dump(account_entities, open(JSON_NAME, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return final_metrics


def cross_validation(dataset, num_epochs=5, folds_number=5, train_proportion=1.0, test_proportion=0.0):
    """ """
    
    iban_list = dataset.AccountNumber.unique()
    isShared = dataset.groupby('AccountNumber', sort=False)['IsShared'].first().loc[iban_list].values
    folds = []
    total_finetuning_metrics = []
    total_clustering_metrics = []

    # Create StratifiedKFold object
    skf = StratifiedKFold(n_splits=folds_number)

    # Iterate through the folds
    for fold, (train_index, test_index) in enumerate(skf.split(iban_list, isShared)):
        print("\n")
        print("--------------------------------------------------------------------------------")
        print(f" ############################## Fold {fold + 1} ############################## ")
        print("--------------------------------------------------------------------------------")
        print("\n")
        print("Train indices:", len(train_index))
        print("Test indices:", len(test_index))
        print("SUM: ", len(train_index) + len(test_index))
        print("TOTAL: ", len(iban_list))
        
        
        # Get the actual AccountNumbers for train and test indices
        X_train = dataset.loc[dataset.AccountNumber.isin(iban_list[train_index])]
        X_test = dataset.loc[dataset.AccountNumber.isin(iban_list[test_index])]
        print("Fine-tuning Training set length  (transactions): ", len(X_train))
        print("Fine-tuning Training set length (ibans)", len(X_train.AccountNumber.unique()))
        print("Proportion on transactions:", len(X_train) / len(dataset))
        print("Proportion on isShared = 1: ", len(X_train.loc[X_train['IsShared'] == 1]) / len(X_train))
        print()
        print("Clustering Testing set length (transactions):", len(X_test))
        print("Clustering Testing set length (ibans)", len(X_test.AccountNumber.unique()))
        print("Proportion on transactions:", len(X_test) / len(dataset))
        print("Proportion on isShared = 1: ", len(X_test.loc[X_test['IsShared'] == 1]) / len(X_test))
        
        f = {}
        f['Train_ibans'] = list(X_train.AccountNumber.unique())
        f['Test_ibans'] = list(X_test.AccountNumber.unique())
        f['Fine-tuning_training_transactions'] = len(X_train)
        f['Fine-tuning_training_ibans'] = len(X_train.AccountNumber.unique())
        f['Fine-tuning_transactions_proportion'] = len(X_train) / len(dataset)
        f['Fine-tuning_isShared_proportion'] = len(X_train.loc[X_train['IsShared'] == 1]) / len(X_train)
        f['Clustering_testing_transactions'] = len(X_test)
        f['Clustering_testing_ibans'] = len(X_test.AccountNumber.unique())
        f['Clustering_transactions_proportion'] = len(X_test) / len(dataset)
        f['Clustering_isShared_proportion'] = len(X_test.loc[X_test['IsShared'] == 1]) / len(X_test)
        folds.append(f)
        
        print("Starting fine tuning...")
        fine_tuning_metrics, model = fine_tuning(X_train, num_epochs=num_epochs, train_proportion=train_proportion, test_proportion=test_proportion)
        print("\n")
        print("Starting clustering...")
        clustering_metrics = clustering(model, X_test)
        print("\n")
        # print("Fine tuning results")
        # for el in fine_tuning_metrics:print("-",el + ":", fine_tuning_metrics[el])
        # print("\nClustering results")
        # for el in clustering_metrics:print(el, clustering_metrics[el])
        
        total_finetuning_metrics.append(fine_tuning_metrics)
        total_clustering_metrics.append(clustering_metrics)
    
    return total_finetuning_metrics, total_clustering_metrics, folds


def export_results(total_finetuning_metrics, total_clustering_metrics, folds):
    """ Refactors and exports results in a json file """
    
    
    final_results = {}
    final_results['fold'] = []
    final_results['train_accuracy_mean'] = 0
    final_results['train_loss_mean'] = 0
    final_results['training_time_mean'] = 0
    # final_results['test_accuracy_mean'] = 0
    # final_results['test_precision_mean'] = 0
    # final_results['test_recall_mean'] = 0
    # final_results['test_f1_mean'] = 0
    
    final_results['isShared_accuracy_mean'] = 0
    final_results['transaction_holder_accuracy_mean'] = 0
    final_results['correct_clustered_task_accuracy_mean'] = 0
    
    for i in range(len(total_finetuning_metrics)):
        fold = {}
        fold['fold'] = folds[i]
        fold['fine_tuning'] = total_finetuning_metrics[i]
        fold['clustering'] = total_clustering_metrics[i]
        final_results['fold'].append(fold)
        
        final_results['train_accuracy_mean'] += total_finetuning_metrics[i]['training_accuracy']
        final_results['train_loss_mean'] += total_finetuning_metrics[i]['training_loss']
        final_results['training_time_mean'] += total_finetuning_metrics[i]['training_time']
        # final_results['test_accuracy_mean'] = 0
        # final_results['test_precision_mean'] = 0
        # final_results['test_recall_mean'] = 0
        # final_results['test_f1_mean'] = 0
        
        final_results['isShared_accuracy_mean'] += total_clustering_metrics[i]['isShared_task']['accuracy']
        final_results['transaction_holder_accuracy_mean'] += total_clustering_metrics[i]['transaction_holder_task']['accuracy']
        final_results['correct_clustered_task_accuracy_mean'] += total_clustering_metrics[i]['correct_clustered_task']['accuracy']
        
    
    final_results['train_accuracy_mean'] = final_results['train_accuracy_mean'] / len(total_finetuning_metrics)
    final_results['train_loss_mean'] = final_results['train_loss_mean'] / len(total_finetuning_metrics)
    final_results['training_time_mean'] = final_results['training_time_mean'] / len(total_finetuning_metrics)
    final_results['isShared_accuracy_mean'] = final_results['isShared_accuracy_mean'] / len(total_clustering_metrics)
    final_results['transaction_holder_accuracy_mean'] = final_results['transaction_holder_accuracy_mean'] / len(total_clustering_metrics)
    final_results['correct_clustered_task_accuracy_mean'] = final_results['correct_clustered_task_accuracy_mean'] / len(total_clustering_metrics)
    
    return final_results        


def main():
    DATE_NAME = str(datetime.now()).split(".")[0].replace(" ", "_") 
    JSON_NAME = "./cross_validation_results/Test_" + DATE_NAME + ".json"
    
    fold_number = 5
    train_proportion = 1.0
    test_proportion = 0.0
    num_epochs = 5
    
    dataset_path = "./Dataset/DATASETPLACEHOLDER"
    
    print("Starting Cross validation...")
    print("Dataset: " + dataset_path)
    print("Number of epochs: " + str(num_epochs))   
    print("Number of folds: " + str(fold_number))
    print("Train proportion (couple prediction): " + str(train_proportion))
    print("Test proportion (couple prediction): " + str(test_proportion))
    print("\n")
    
    dataset = pd.read_csv(dataset_path)
    total_finetuning_metrics, total_clustering_metrics, folds = cross_validation(dataset, num_epochs, fold_number, train_proportion, test_proportion)
    print("End Cross validation! ")
    
    final_results = export_results(total_finetuning_metrics, total_clustering_metrics, folds)
    final_results['num_epochs'] = num_epochs
    final_results['batch_size'] = batch_size
    final_results['weight_decay'] = weight_decay
    final_results['learning_rate'] = learning_rate
    final_results['train_proportion'] = train_proportion
    final_results['test_proportion'] = test_proportion
    final_results['dataset_path'] = dataset_path
    
    json.dump(final_results, open(JSON_NAME, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    print("\nExporting results in: " + JSON_NAME)
    

if __name__ == "__main__":
  main()
