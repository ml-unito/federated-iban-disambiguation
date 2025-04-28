import os
from lib.datasetManipulation import labeled_pairs
import lib.string_kernels as sk
from federated_learning import create_couple_df
from rich.console import Console
import pandas as pd

console = Console()


def load_data(train_path: str, test_path: str):
    cp_train_df = pd.read_csv(train_path)
    cp_test_df = pd.read_csv(test_path)

    train_pairs, train_labels = labeled_pairs(cp_train_df)
    test_pairs, test_labels = labeled_pairs(cp_test_df)

    train_xy = list(zip(train_pairs, train_labels))
    test_xy = list(zip(test_pairs, test_labels))

    console.log(f"train xy: {len(train_xy)}")
    console.log(f"test xy: {len(test_xy)}")
    console.log(f"train xy head: {train_xy[:5]}")
    console.log(f"test xy head: {test_xy[:5]}")

    # Create a dataframe from the train and test data, split the pairs into two columns
    train_xy = [[pair[0], pair[1], label] for pair, label in train_xy]
    train_df = pd.DataFrame(train_xy, columns=["str1", "str2", "label"])

    test_xy = [[pair[0], pair[1], label] for pair, label in test_xy]
    test_df = pd.DataFrame(test_xy, columns=["str1", "str2", "label"])

    return train_df, test_df


def save_data(fname, data):
    sims =  [[sk.spectrum_kernel([s1],[s2],p=4)[0].item(), label] for s1,s2,label in data.itertuples(index=False)]

    with open(fname, "w") as f:
        f.write("similarity,label\n")
        for sim in sims:
            f.write(f"{sim[0]},{sim[1]}\n")
    
def data_already_saved():
    if os.path.exists("dataset/similarity_train.csv") and os.path.exists("dataset/similarity_test.csv"):
        return True
    
    return False

def load_sim_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


if not data_already_saved():
    console.log("Data not saved, creating it")

    train,test = load_data(
        train_path="dataset/split_dataset/df_train.csv",
        test_path="dataset/split_dataset/df_test.csv"
    )

    if not os.path.exists("dataset/similarity_train.csv"):
        console.log("Saving train data")
        save_data("dataset/similarity_train.csv", train)

    if not os.path.exists("dataset/similarity_test.csv"):
        console.log("Saving test data")
        save_data("dataset/similarity_test.csv", test)
else:
    console.log("Data already created")

train, test = load_sim_data(
    train_path="dataset/similarity_train.csv",
    test_path="dataset/similarity_test.csv"
)

console.log(f"train: {train.head()}")
console.log(f"test: {test.head()}")

console.log(f"min train: {train['similarity'].min()}")
console.log(f"max train: {train['similarity'].max()}")
console.log(f"min test: {test['similarity'].min()}")
console.log(f"max test: {test['similarity'].max()}")

train_max = int(train['similarity'].max())

fps = []
tps = []

fps_test = []
tps_test = []

accuracy_train = []
accuracy_test = []

for threshold in range(train_max+1):
    console.log(f"Threshold: {threshold}")
    predicted_train = train['similarity'].apply(lambda x: 1 if x <= threshold else 0)
    predicted_test = test['similarity'].apply(lambda x: 1 if x <= threshold else 0)

    # print confusion matrix using sklearn
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score

    train_labels = train['label']
    test_labels = test['label']
    train_confusion_matrix = confusion_matrix(train_labels, predicted_train)
    test_confusion_matrix = confusion_matrix(test_labels, predicted_test)

    fps.append(train_confusion_matrix[0][1])
    tps.append(train_confusion_matrix[1][1])

    fps_test.append(test_confusion_matrix[0][1])
    tps_test.append(test_confusion_matrix[1][1])

    accuracy_score_train = accuracy_score(train_labels, predicted_train)
    accuracy_score_test = accuracy_score(test_labels, predicted_test)

    accuracy_train.append(accuracy_score_train)
    accuracy_test.append(accuracy_score_test)


print("Max accuracy train: ", max(accuracy_train))
print("Max accuracy test: ", max(accuracy_test))
print("Max accuracy train index: ", accuracy_train.index(max(accuracy_train)))
print("Max accuracy test index: ", accuracy_test.index(max(accuracy_test)))

threshold = accuracy_train.index(max(accuracy_train))
predicted_train = train['similarity'].apply(lambda x: 1 if x <= threshold else 0)
predicted_test = test['similarity'].apply(lambda x: 1 if x <= threshold else 0)
confusion_matrix_train = confusion_matrix(train_labels, predicted_train)
confusion_matrix_test = confusion_matrix(test_labels, predicted_test)
print("Train confusion matrix: ", confusion_matrix_train)
print("Test confusion matrix: ", confusion_matrix_test)


from matplotlib import pyplot as plt
plt.plot(fps, tps, label="train")    
plt.xlabel("False Positives")
plt.ylabel("True Positives")
plt.title("False Positives vs True Positives")
plt.legend()
plt.savefig("dataset/false_positives_vs_true_positives.png")
plt.clf()

plt.plot(fps_test, tps_test, label="test")
plt.xlabel("False Positives")
plt.ylabel("True Positives")
plt.title("False Positives vs True Positives")
plt.legend()
plt.savefig("dataset/false_positives_vs_true_positives_test.png")
plt.clf()

plt.plot(range(train_max+1), accuracy_train, label="train")
plt.plot(range(train_max+1), accuracy_test, label="test")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Threshold")
plt.legend()
plt.savefig("dataset/accuracy_vs_threshold.png")


