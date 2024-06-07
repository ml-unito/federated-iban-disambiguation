import json
import random
import pandas as pd
from lib.plot import *
from lib.saveOutput import *
from lib.datasetManipulation import *
from lib.download import download_fine_tuned_model
from torch.utils.data import TensorDataset
from lib.CharacterBertForClassification_v1 import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


download_fine_tuned_model()
model = CharacterBertForClassification()
tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert')
model.to(device)


# test name
DATE_NAME = str(datetime.now()).split(".")[0].replace(" ", "_") 
LOG_NAME = ("log_" + DATE_NAME + ".txt").replace(":", "_")
PLOT_NAME = ("CM_matrix_" + DATE_NAME + ".png").replace(":", "_")
#DATASET_BUILD = "output_dataset_" + DATE_NAME + ".csv"

# New log File
saveToFile = SaveOutput('./Test/', LOG_NAME, printAll=True)
with open('./config/parameters.json', "r") as data_file:
    parameters = json.load(data_file)


# New log File
with open('./config/group_parameters.json', "r") as data_file:
    group_parameters = json.load(data_file)


# parameters
batch_size = parameters['batch_size']
grouping_placheholder = group_parameters['grouping_placheholder']
grouping_padding = group_parameters['grouping_padding']
removepunctuation = group_parameters['remove_punctuation']



def preprocess_dataset(dataset):
    """ """
    preprocessDataset = dataset.drop("AccountNumber", axis=1)
    df = pd.DataFrame()
    df['text'] = preprocessDataset.drop("IsShared",axis=1).apply(lambda row: ' '.join(row.astype(str)), axis=1)
    df['label'] = preprocessDataset['IsShared']
    return df



def tokenize_dataset(dataframe):
    """ Tokenize dataset """
    return dataframe['text'].apply(lambda x: ['[CLS]', *tokenizer.basic_tokenizer.tokenize(x), '[SEP]'] )
    


def lookup_table(tokenized_texts, dataframe):
    """ define the input tensors"""
    indexer = CharacterIndexer()
    input_tensors = []
    input_tensors = indexer.as_padded_tensor(tokenized_texts)     # Create input tensor
    labels = torch.tensor(dataframe['label'].values)              # Create labels tensor
    return input_tensors, labels


def rebuild_dataset_for_pretty_print(dataset, predicted, name):
    """ """
    path = "./output_build_dataset/" + name
    dataset['predicted'] = predicted
    save_dataset_csv(dataset, path)



def print_notes():
    """ Pretty print notes """

    print()
    print("-------------------------------------------------------")
    print("----------- UNITO ENTITY RECOGNITION TEST -------------")
    print("-------------------------------------------------------")
    print()

    print("--------------------EXAMPLE USAGE-----------------------")
    print("----- python3 unitoBert.py input/dataset_100_1.csv -----")
    print("--------------------------------------------------------")
    print



def main():
    """ """
    
    print_notes()
    if len(sys.argv) < 2:
        print("\nType model and datasets, first!")
        print("USAGE: python unitoBert.py DATASET_PATH")
        exit()

    saveToFile("Output Log:  " + LOG_NAME + "\n")
    saveToFile("Dataset loaded...\n")
    saveToFile("\n\nModel: ")
    
    try:
        out = model.load_state_dict(torch.load("./character_bert_model/pretrained-models/general_character_bert/pytorch_model.bin"))
    except Exception:
        out = model.load_state_dict(torch.load("./character_bert_model/pretrained-models/general_character_bert/pytorch_model.bin", map_location ='cpu'))
    
    saveToFile(str(out))
    datasetPath = sys.argv[1]
    if ".csv" in datasetPath: dataset = load_dataset_csv(datasetPath)
    else: dataset = load_dataset_excel(datasetPath)

    dataset = group_dataset(dataset, grouping_padding, placeholder=grouping_placheholder)
    saveToFile("Dataset groupped ...\n")
    saveToFile("Dataset and model loaded...\n")
    saveToFile("Dataset Preview\n")
    saveToFile(dataset[dataset.columns[:3]].head(5).to_markdown())

    saveToFile("\nBalanced dataset, IsShared statistics")
    saveToFile(str(dataset.groupby('IsShared').size()))
    
    dataframe = preprocess_dataset(dataset)
    tokenized_texts = tokenize_dataset(dataframe)
    input_tensors, labels = lookup_table(tokenized_texts,dataframe)
    data = TensorDataset(input_tensors, labels)

    # Create the DataLoaders for our training and validation sets.
    dataloader = DataLoader(
                data,                                        # The training samples.
                sampler = SequentialSampler(data),           # Select batches randomly
                batch_size = batch_size                      # Trains with this batch size.
            )

    # Free memory
    del dataframe
    del input_tensors
    del labels
    
    criterion = nn.BCELoss()
    _, metrics, predictions, total_labels = evaluate(model, dataloader, criterion)
    saveToFile("Testing on test set...")
    saveToFile("")
    for el in metrics: saveToFile(el +  ":" + str(metrics[el]))

    plot_confusion_matrix(total_labels, predictions, ['Class 0', 'Class 1'], (7,4), saveName=PLOT_NAME) 
    #rebuild_dataset_for_pretty_print(dataset, predictions, DATASET_BUILD)

main()