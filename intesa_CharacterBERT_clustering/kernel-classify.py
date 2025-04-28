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

def get_features_from_pairs(s1, s2, n_features=4):
    return [
        sk.spectrum_kernel([s1], [s2], p=i)[0].item() for i in range(1, n_features + 1)
    ]

def save_data(fname, data, oversample=False):
    n_features = 7
    sims = [ get_features_from_pairs(s1, s2, n_features=n_features) + [label] for s1, s2, label in data.itertuples(index=False)]
    
    if oversample:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X = [s[:n_features] for s in sims]
        y = [s[n_features] for s in sims]

        console.log(f"Oversampling data")
        X_res, y_res = sm.fit_resample(X, y)
        sims = [[*s[:n_features], label] for s, label in zip(X_res, y_res)]

    with open(fname, "w") as f:
        f.write(f"{','.join([f'p{i}' for i in range(1,n_features+1)])},label\n")
        for sim in sims:
            f.write(f",".join([str(s) for s in sim]) + "\n")
    
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
        save_data("dataset/similarity_train.csv", train, oversample=True)

    if not os.path.exists("dataset/similarity_test.csv"):
        console.log("Saving test data")
        save_data("dataset/similarity_test.csv", test)
else:
    console.log("Data already created")

train, test = load_sim_data(
    train_path="dataset/similarity_train.csv",
    test_path="dataset/similarity_test.csv"
)

# import logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# normalize using minmax scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train.iloc[:, :-1] = scaler.fit_transform(train.iloc[:, :-1])
test.iloc[:, :-1] = scaler.transform(test.iloc[:, :-1])


lr = LogisticRegression(max_iter=1000)
train_x = train.iloc[:, :-1].values
train_y = train.iloc[:, -1].values
test_x = test.iloc[:, :-1].values
test_y = test.iloc[:, -1].values
lr.fit(train_x, train_y)
predictions = lr.predict(test_x)
console.log(f"Accuracy: {accuracy_score(test_y, predictions)}")
console.log(f"Classification report:\n {classification_report(test_y, predictions)}")

console.log(f"Coefs: {lr.coef_}")
console.log(f"Intercept: {lr.intercept_}")