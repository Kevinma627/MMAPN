'''
TabNet uses PyTorch 1.13.1
Ensure this is being run in an environment with the correct dependency versions for this version of PyTorch
(May conflict with dependency versions for Tensorflow)
'''
import argparse
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="Train TabNet model")

parser.add_argument('--test_data_file', type=str, default='Test_Data/test_tab.csv', help='Test data file')
parser.add_argument('--model_save_dir', type=str, default='Model/runs_tab', help="Where the model is saved")
parser.add_argument('--model_name', type=str, default='MMAPN_tab', help="Name of the model")

args = parser.parse_args()

# Read in values and labels
df = pd.read_csv(args.test_data_file)
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Load model
clf = TabNetClassifier()
clf.load_model(args.model_save_dir + "/" + args.model_name + ".zip")

preds = clf.predict(X)
probs = clf.predict_proba(X)
confidences = np.max(probs, axis=1) 
accuracy = accuracy_score(Y, preds)
print(f'Test Accuracy: {accuracy}')

# Get feature importances (test)
M_explain, aggregated_mask = clf.explain(X)

feature_importances = np.sum(np.abs(M_explain), axis=0)
feature_importances = feature_importances / np.sum(feature_importances) # Normalize

# Save results
out_file = args.model_save_dir + "/test_results.txt"

with open(out_file, 'w') as fout:
    fout.write("label\tpredict\tconfidence\tAccuracy: " + str(accuracy) +"\n")
    for true, pred, conf in zip(Y, preds, confidences):
        fout.write(f"{true}\t{pred}\t{conf}\n")
    
    fout.write("\nFeature Importances: \n")
    for feature_index, importance_score in enumerate(feature_importances):
        fout.write(f"Feature {feature_index}: {importance_score:.4f}\n")