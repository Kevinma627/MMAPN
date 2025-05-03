'''
TabNet uses PyTorch 1.13.1
Ensure this is being run in an environment with the correct dependency versions for this version of PyTorch
(May conflict with dependency versions for Tensorflow)
'''
import argparse
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Train TabNet model")

parser.add_argument('--train_data_file', type=str, default='Train_Data/domain_data.csv', help='Train data file')
parser.add_argument('--data_dev_pct', type=float, default=0.1, help="Percentage of data used for validation")
parser.add_argument('--model_save_dir', type=str, default='Model/runs_tab', help="Where the model saves")
parser.add_argument('--model_name', type=str, default='MMAPN_tab', help="Name of the model")

args = parser.parse_args()

# Read in values and labels
df = pd.read_csv(args.train_data_file)
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

X_train, X_val, Y_train, y_val = train_test_split(X, Y, test_size=args.data_dev_pct, random_state=42, shuffle=True, stratify=Y)

# Train the model
clf = TabNetClassifier(device_name="cuda")
clf.fit(
    X_train, Y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=["auc", "accuracy", "logloss", "balanced_accuracy"] # Last metric is used for earl stopping
)

# Save the model
saving_path_name = args.model_save_dir + "/" + args.model_name
saved_filepath = clf.save_model(saving_path_name)

# Save feature importance (train)
with open(args.model_save_dir + "/feature_importance.txt", 'w') as file:
    for feature_index, importance_score in enumerate(clf.feature_importances_):
        file.write(f"Feature {feature_index}: {importance_score:.4f}\n")
