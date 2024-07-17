import json
import logging
import pandas as pd
from lib.plot import *
from lib.saveOutput import *
from itertools import combinations
from lib.datasetManipulation import *
from torch.utils.data import TensorDataset
from lib.CharacterBertForClassification import *
from lib.download import download_fine_tuned_model
from torch.utils.data import DataLoader, SequentialSampler
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# Load Custom model
download_fine_tuned_model()
model = CharacterBertForClassification()
model.to(device)

# test name
DATE_NAME = str(datetime.now()).split(".")[0].replace(" ", "_").replace(":","_")
LOG_NAME = "link_prediction_test_log_" + DATE_NAME + ".txt"
PLOT_NAME = "./Plot/link_prediction_CM_matrix_" + DATE_NAME + ".png"
JSON_NAME = "./Clusters/Clusters_" + DATE_NAME + ".json"
DATASET_BUILD = "./Rebuild_dataset/labelled_dataset_" + DATE_NAME + ".csv"


# debug mode
DEBUG_MODE = False

# New log File
saveToFile = SaveOutput('./Log/', LOG_NAME, printAll=True, debug=DEBUG_MODE)

# parameters
batch_size = 32


def create_pairs(dataset):
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


def prepocess_dataset(dataset):
    """ """
    
    if "Address" in dataset.columns: 
        dataset = dataset.drop("Address", axis=1)
    return dataset.drop_duplicates()



def tokenize_dataset(dataframe):
    """ Tokenize dataset """
    return dataframe['text'].apply(lambda x: ['[CLS]', *x.split("@"), '[SEP]'] )
    


def lookup_table(tokenized_texts, dataframe):
    """ define the input tensors"""
    indexer = CharacterIndexer()
    input_tensors = []
    input_tensors = indexer.as_padded_tensor(tokenized_texts)     # Create input tensor
    labels = torch.tensor(dataframe['label'].values)              # Create labels tensor
    return input_tensors, labels



def main():
    """ """

    if len(sys.argv) < 2:
        print("\nType model and datasets, first!")
        print("USAGE: python unitoBertNer DATASET_PATH")
        exit()

    # load dataset
    datasetPath = sys.argv[1]
    saveToFile("Output Log " + str(datetime.now()) + "\n")
    saveToFile("Dataset path: " + datasetPath)
    saveToFile("Model path: " + sys.argv[1])   
    saveToFile("Loading dataset and model...")
    saveToFile("Dataset loaded...\n")
    
    # load model
    model_path = "./character_bert_model/pretrained-models/general_character_bert/pytorch_model.bin"
    saveToFile("\n\nModel: ")
    try:
        out = model.load_state_dict(torch.load(model_path))
    except:
        out = model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
    saveToFile(str(out))
    if ".csv" in datasetPath: dataset = load_dataset_csv(datasetPath)
    else: dataset = load_dataset_excel(datasetPath)

    # print dataset
    saveToFile("Dataset and model loaded...\n")
    saveToFile("Dataset Preview\n")
    saveToFile(dataset.head(5).to_markdown())
    
    
    # preprocess dataset
    saveToFile("Pairing dataset...")
    dataset = prepocess_dataset(dataset)
    #dataset.to_csv(DATASET_TEST, index=False)
    saveToFile("\ndataset, IsShared statistics")
    saveToFile(str(dataset.groupby('IsShared').size()))
    
    
    # create pairs
    dataframe = create_pairs(dataset)
    saveToFile("\nDataset Preview\n")
    saveToFile(dataframe.drop("text",axis=1).head(30).to_markdown())
    saveToFile("\ndataset, Label statistics")
    saveToFile(str(dataframe.groupby('label').size()))
    saveToFile("\nPreprocessed info:")
    saveToFile(dataframe['text'][0])
    saveToFile("")
    
    
    # tokenize pairs
    tokenized_texts = tokenize_dataset(dataframe)
    dataframe = dataframe.drop('text', axis=1)
    saveToFile("\nTokenized text:")
    saveToFile(str(tokenized_texts[:10]))
    saveToFile("")
    input_tensors, labels = lookup_table(tokenized_texts,dataframe)
    data = TensorDataset(input_tensors, labels)


    # Create the DataLoaders for load the test set
    dataloader = DataLoader(
                data,                                        # The training samples.
                sampler = SequentialSampler(data),           # Select batches sequentially
                batch_size = batch_size                      # Trains with this batch size.
            )

    
    # Free memory
    del tokenized_texts
    del input_tensors
    del labels


    # Evaluate model
    saveToFile("Evaluation of the model on test set on the differency_couples task...")
    criterion = nn.BCELoss()
    _, metrics, predictions, total_labels = evaluate(model, dataloader, criterion)
    # saveToFile("Link prediction task metrics:")
    # for el in metrics: saveToFile("- Link prediction - " + el +  ":" + str(metrics[el]))
    #if not DEBUG_MODE: plot_confusion_matrix(total_labels, predictions, ['Same name (0)', 'Different name(1)'], (7,4), saveName=PLOT_NAME) 
    
    
    # print results
    dataframe['predicted'] = predictions
    saveToFile("\nDataset Preview\n")
    saveToFile(dataframe.head(30).to_markdown())
    saveToFile("")
    #if not DEBUG_MODE: dataframe.to_csv(DATASET_BUILD, index=False)
    
    
    # Free memory
    del dataloader
    del data
    del total_labels
    
    
    account_entities = {}
    dataframe_groupped = dataframe.groupby("iban")
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
        
        # Add edges based on predictions
        for i in range(len(predicted)):
            if predicted[i] == 0: G.add_edge(names1[i], names2[i])

        if len(predicted) != len(names1):
            print(iban, len(predicted), len(names1), len(names2))

        clusters = list(nx.connected_components(G))
        for cluster in clusters:
            cluster_list = list(cluster)
            representative_name = max(cluster, key=len)
            
                
            r_nodes = [el for el in cluster_list]
            account_entities[iban]['holders'].append({
                    "cluster_name": representative_name,
                    "names_list": r_nodes,
                    "holder_from_cluster_name": dataset[dataset['Name'] == representative_name]['Holder'].tolist()[0],
            })
        
    # IsShared Task Accuracy
    for a in account_entities:
        if len(account_entities[a]['holders']) > 1: account_entities[a]['predicted_shared'] = 1
        elif len(account_entities[a]['holders']) == 1: account_entities[a]['predicted_shared'] = 0
            
    
    predictions = [account_entities[el]['predicted_shared'] for el in account_entities]
    real = [account_entities[el]['IsShared'] for el in account_entities]
        
    count = 0
    number_cluster_ok = 0
    for a in account_entities:
        if account_entities[a]['IsShared'] == account_entities[a]['predicted_shared']:
            count += 1
        
        holders = []
        for h in account_entities[a]['holders']:holders.append(h['holder_from_cluster_name'])
        if set(account_entities[a]['real_holders']) == set(holders):
            number_cluster_ok += 1
    
    
    # Transaction Holder Accuracy
    dataset['Predicted_Holder'] = ["" for el in range(len(dataset))]
    for iban in account_entities:
        holder_dict = {}
        for holder in account_entities[iban]['holders']:
            for name in holder["names_list"]:       
                holder_dict[name] = holder['holder_from_cluster_name']
        
        for index, row in dataset.loc[dataset['AccountNumber'] == iban].iterrows():
            dataset.loc[index,"Predicted_Holder"] = holder_dict[row['Name']]
        
    number_transaction_ok = len(dataset.loc[dataset['Holder'] == dataset['Predicted_Holder']])
    
    
    # Print all
    saveToFile("Evaluation of the model on the IsShared classification task...")
    saveToFile("Number prediction IsShared OK: " + str(count))
    saveToFile("Number of iban: " + str(len(dataframe_groupped)))    
    #saveToFile("IsShared Accuracy: " + str(count / len(dataframe_groupped)))
    metrics = compute_metrics(predictions, real)
    if not DEBUG_MODE: plot_confusion_matrix(real, predictions, ['Is Shared', 'Is not Shared'], (7,4), saveName=PLOT_NAME) 
    
    
    
    for el in metrics: saveToFile("- " + el +  ":" + str(metrics[el]))
    saveToFile("")
    
    saveToFile("\n")
    saveToFile("Evaluation of the model on the correct Holder prediction...")
    saveToFile("Number of holders exactly predicted: " + str(number_cluster_ok))
    saveToFile("Number of iban: " + str(len(dataframe_groupped)))    
    saveToFile("- Holder Accuracy: " + str(number_cluster_ok / len(dataframe_groupped)))
    saveToFile("")
    
    saveToFile("\n")
    saveToFile("Evaluation of the model on the correct transaction prediction...")
    saveToFile("Number of transaction exactly predicted: " + str(number_transaction_ok))
    saveToFile("Number of transaction:" + str(len(dataset)))    
    saveToFile("- Transaction Holder Accuracy:" + str(number_transaction_ok / len(dataset)))
    saveToFile("")
    
    
    # save clusters on json file
    saveToFile("Exporting clusters on json file...")
    if DEBUG_MODE: json.dump(account_entities, open("./Clusters/clusters_test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    else: json.dump(account_entities, open(JSON_NAME, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    
main()