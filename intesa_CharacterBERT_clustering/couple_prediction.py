import os
import json
import time
from lib.plot import *
from lib.saveOutput import *
from collections import Counter
from lib.datasetManipulation import *
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from lib.download import download_pre_trained_model

download_pre_trained_model()
from lib.CharacterBertForClassificationOptimized import *


# Load Custom model
model = CharacterBertForClassification()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


# test name
DEBUG = False
DATE_NAME = str(datetime.now()).split(".")[0].replace(" ", "_") 
LOG_NAME = "train_log_" + DATE_NAME + ".txt"
PLOT_NAME = "./couple_prediction/Plot/plot_train/plot_" + DATE_NAME + ".png"
BEST_MODEL_PATH = './couple_prediction/Output_model/model_weight_' + DATE_NAME + '.pt'


# New log File
saveToFile = SaveOutput('./couple_prediction/Log/', LOG_NAME, printAll=True, debug=DEBUG)
with open('./config/parameters.json', "r") as data_file:
    parameters = json.load(data_file)



# Retrieve parameters
train_proportion = parameters['train_proportion']
val_proportion = parameters['val_proportion']
test_proportion = parameters['test_proportion']
batch_size = parameters['batch_size']
num_epochs = parameters['num_epochs']
weight_decay = parameters['weight_decay']
learning_rate = parameters['learning_rate']




def main():
    """ Main function 
    - Read dataset
    - Prepocess dataset
    - Tokenize dataset
    - Train model
    - Save model
    
    The dataset is a .csv or .xlsx file. From an existing dataset the program 
    generate another dataset with the couples Name1 - Name2 - Label. 
    On this new dataset the model is trained to recognize the couples.
    """
    
    if len(sys.argv) < 2:
        print("\nType a dataset, first!")
        print("USAGE: python3 couple_prediction.py DATASET_PATH BALANCE")
        print("where, DATASET_PATH is a .csv or .xlsx file")
        print("where, BALANCE for balancing the dataset. Take one of ['unbalance' or 'balance'] default balance")
        exit()

    balance = True
    if len(sys.argv) == 3:
        if(sys.argv[2] == "unbalance"):
            balance = False
    

    # -------------------------------------------------------
    # Print some informations
    # -------------------------------------------------------

    saveToFile("")
    for el in parameters: saveToFile("- " +str(el) + " - " + str(parameters[el]))
    saveToFile("\n\nModel:\n")
    saveToFile(str(model))
    saveToFile("\n")
    


    # -------------------------------------------------------
    # Load the dataset and print the preview on the log file
    # -------------------------------------------------------
    
    saveToFile("Output Log: " + str(datetime.now()))
    datasetPath = sys.argv[1]
    datasetName = os.path.basename(datasetPath).split(".")[0]
    dataset_directory_path = "./Dataset/Train/" + datasetName
    dataset_preprocessed_path = dataset_directory_path + "/" + datasetName + "_dataset.csv"
    if not os.path.exists(dataset_directory_path):os.makedirs(dataset_directory_path)
    
    dataset = load_dataset(datasetPath)
    saveToFile("\nDataset path: " + datasetPath)
    saveToFile("Dataset Preview\n")
    saveToFile(dataset.head(5).to_markdown())
    
    
    
    # -------------------------------------------------------
    # Preprocessing dataset. The preprocessing remove the "address" column
    # Eventually - ballancing the dataset on the IsShared column
    # -------------------------------------------------------
    
    dataset = prepocess_dataset(dataset)
    if balance:
        dataset = balance_dataset(dataset,"IsShared")
        saveToFile("\nBalancing the dataset on the isShared column")
        saveToFile("Dataset, isShared statistics:")
        saveToFile(str(dataset.groupby('IsShared').size()))
    else:
        saveToFile("\nDataset, isShared statistics")
        saveToFile(str(dataset.groupby('IsShared').size()))
            
    
    
    # -------------------------------------------------------
    # Create the dataset of pairs for the couple prediction task
    # Eventually - ballancing the dataset on the label column
    # -------------------------------------------------------
    
    dataframe = create_pairs(dataset)
    saveToFile("Preview of the dataset for the couple prediction taks:\n")
    saveToFile(dataframe.head(10).to_markdown())
    
    if balance:
        dataframe = balance_dataset(dataframe, "label")
        saveToFile("\n\nBalancing the new dataset on the labels column")
        saveToFile("Dataset, label statistics")
        saveToFile(str(dataframe.groupby('label').size()))
    else:
        saveToFile("\nDataset, label statistics")
        saveToFile(str(dataframe.groupby('label').size()))
    
    
    
    # -------------------------------------------------------
    # Splitting dataset
    # -------------------------------------------------------
    
    saveToFile("\n\n- Preview of the dataset before the tokenization step:")
    saveToFile(dataframe['text'].head(5).to_markdown())
    saveToFile("")
    
    dataframe.to_csv(dataset_preprocessed_path, index=False)
    X = tokenize_dataset(dataframe).tolist()
    y = dataframe['label'].tolist()
    saveToFile("\n- Preview of the dataset after the tokenization step:")
    for i in range(5):saveToFile(str(X[i]))
    saveToFile("")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, random_state=42, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, random_state=42, stratify=y_test)
    

    # Print the class distribution
    saveToFile("Original data: " + str(Counter(y)))
    saveToFile("Training data: " + str(Counter(y_train)))
    saveToFile("Test data: " + str(Counter(y_test)))
    saveToFile("Validation data: " + str(Counter(y_val)))
    saveToFile("")

    saveToFile("Training set size: " + str(len(X_train)))
    saveToFile("Validation set size: " + str(len(X_val)))
    saveToFile("Test set size: " + str(len(X_test)))
    saveToFile("")

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
    saveCriteria = SaveBestModel(BEST_MODEL_PATH, 0.015)
    saveToFile("\n\nTraining step")
    init = time.time()
    
    
    for epoch in range(num_epochs):
       
        # --------------------------------------
        # Training step
        # --------------------------------------
        
        saveToFile("\n\n------ Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + " ------\n")
        loss_t, accuracy_t = train(model, X_train, y_train, batch_size, optimizer, criterion, scheduler)
        training_loss.append(loss_t)
        training_accuracy.append(accuracy_t)

        
        # --------------------------------------
        # Evaluate on validation set
        # --------------------------------------
        
        loss_v, metrics, _, _ = test(model, X_val, y_val, batch_size, criterion)
        validation_loss.append(loss_v)
        validation_accuracy.append(metrics["accuracy"])
        validation_f1.append(metrics['f1'])
        saveCriteria(model, (epoch + 1), loss_t, loss_v)
        
        saveToFile(f'- Training set loss: {loss_t}')
        saveToFile(f'- Training set accuracy: {accuracy_t} \n')
        saveToFile(f'- Validation set loss: {loss_v}')
        for el in metrics: saveToFile("- " + el +  ":" + str(metrics[el]))
        saveToFile(" ")
        
        
        # Evaluate The early stopping
        if early_stopping_train(loss_t) or early_stopping_val(loss_v):
            saveToFile("\nEarly stopping after {} epochs!".format(epoch + 1))
            break
    
    end = time.time()
    saveToFile("\n\nTraining time: " + str((end - init) / 60) + " minutes")
    
    
    
    
    # --------------------------------------
    # Evaluate on test set
    # --------------------------------------

    saveToFile("\nTesting on Test Set\n")
    _, metrics, predictions, total_labels = test(model, X_test, y_test, batch_size, criterion)
    for el in metrics: saveToFile("- " + el +  ":" + str(metrics[el]))
    if not DEBUG: plot_confusion_matrix(total_labels, predictions, ['Same name (0)', 'Different name(1)'], (7,4), saveName=PLOT_NAME) 
    
    
    # -------------------------------------------------------
    # Plot the metrics
    # -------------------------------------------------------
    
    plot_metrics(training_loss, validation_loss, validation_accuracy, validation_f1, (9,4), saveName=PLOT_NAME)
    saveToFile("\nYou can see the plot at the link: " + PLOT_NAME)
    saveToFile("You can see the model .pt file at the link: " + BEST_MODEL_PATH)
    
    

main()