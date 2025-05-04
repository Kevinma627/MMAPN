import argparse
from utils import get_results

parser = argparse.ArgumentParser(description="Stacking Script")
parser.add_argument('--url_save_dir', type=str, default="Model/runs_url", help="URL model save directory")
parser.add_argument('--html_save_dir', type=str, default="Model/runs_html", help="HTML model save directory")
parser.add_argument('--tab_save_dir', type=str, default="Model/runs_tab", help="TabNet model save directory")
parser.add_argument('--results_dir', type=str, default="Final_results", help="Where to save results")
args = parser.parse_args()

# Prediction thresholds for URL and HTML models (Should match the thresholds from test_url.py and test_html.py)
URL_THRESH = 0.72
HTML_THRESH = 0.54

# Gets the accuracies from each individual sub-model
def accuracy_get():
    with open(args.url_save_dir + "/test_results.txt", "r") as file:
        line = file.readline().strip()
        _, acc = line.rsplit("Accuracy:", 1)
        url_acc = float(acc)
    with open(args.html_save_dir + "/test_results.txt", "r") as file:
        line = file.readline().strip()
        _, acc = line.rsplit("Accuracy:", 1)
        html_acc = float(acc)
    with open(args.tab_save_dir + "/test_results.txt", "r") as file:
        line = file.readline().strip()
        _, acc = line.rsplit("Accuracy:", 1)
        tab_acc = float(acc)

    return url_acc, html_acc, tab_acc

# Get the results and min/max scores from the sub-model runs
url_results, url_min, url_max = get_results(args.url_save_dir + "/test_results.txt", False)
html_results, html_min, html_max = get_results(args.html_save_dir + "/test_results.txt", False)
tab_results, tab_min, tab_max = get_results(args.tab_save_dir + "/test_results.txt", True)

scores = []
conf_urls = []
conf_htmls = []
conf_tabs = []
preds = []
labels = []

# ---------------------------------------------Weighted-Mean stacking function---------------------------------------------
'''
- For every entry
    For URL and HTML results:
    - Calculates the score deviation from the threshold (pos for mal and neg for benign)
    - Calculates the range of scores for both classes
    - Calculates confidence by doing deviation/range (will be pos or neg depending on predicted class)
    For Tab results:
    - Calculates the range of confidences
    - Normalizes the confidences by doing (confidence - min)/range
    - Changes sign of confidence depending on predicted class
- Skips calculations for HTML model if no HTML results exist (because no HTML content could be fetched)
- Calculates final score by adding confidences of sub-models (skips HTML confidence if it was not calculated)
- Makes a final prediction from that score (based on sign)
'''
for url, html, tab in zip(url_results, html_results, tab_results):
    thresh_deviate = url[2] - URL_THRESH

    if thresh_deviate > 0:
        confidence_url = thresh_deviate/(url_max - URL_THRESH)
    else:
        confidence_url = thresh_deviate/(URL_THRESH - url_min)
    
    if html[2] != None:
        thresh_deviate = html[2] - HTML_THRESH

        if thresh_deviate > 0:
            confidence_html = thresh_deviate/(html_max - HTML_THRESH)
        else:
            confidence_html = thresh_deviate/(HTML_THRESH - html_min)
    else:
        confidence_html = None

    r = tab_max - tab_min

    if tab[1] == 0:
        confidence_tab = -(tab[2] - tab_min)/r
    else:
        confidence_tab = (tab[2] - tab_min)/r

    if confidence_html != None:
        final_score = confidence_url + confidence_html + confidence_tab
    else:
        final_score = confidence_url + confidence_tab

    if final_score > 0:
        preds.append(1)
    else:
        preds.append(0)

    conf_urls.append(confidence_url)
    conf_htmls.append(confidence_html)
    conf_tabs.append(confidence_tab)
    scores.append(final_score)
    labels.append(int(tab[0])) # Get true labels from one of the model results

# Calc accuracy
correct = 0
for label, pred in zip(labels, preds):
    if label == pred:
        correct += 1

accuracy_mean = correct/len(labels)

# Save final results
with open(args.results_dir + "/WeightedMean.txt", 'w') as file:
    file.write("label\tpredict\tscore\turl_confidence\thtml_confidence\ttab_confidence\tAccuracy: " + str(accuracy_mean) + "\n")
    for label, pred, score, url, html, tab in zip(labels, preds, scores, conf_urls, conf_htmls, conf_tabs):
        file.write(f"{label}\t{pred}\t{score}\t{url}\t{html}\t{tab}\n")

# ---------------------------------------------Highest-Confidence stacking function---------------------------------------------
'''
- For every entry:
    - Finds the largest confidence (absolute value)
    - Matches that confidence to its respective model
    - Prediction == the prediction of the matched model
- If no html model results exist: leaves html model out of prediction consideration
'''

preds = [] # Reset predictions

for url, html, tab in zip(conf_urls, conf_htmls, conf_tabs):
    if html != None:
        highest = max(abs(url), abs(html), abs(tab))
    else:
        highest = max(abs(url), abs(tab))
        
    if highest == abs(url):
        if url > 0:
            preds.append(1)
        else:
            preds.append(0)
    elif html != None and highest == abs(html):
        if html > 0:
            preds.append(1)
        else:
            preds.append(0)
    else:
        if tab > 0:
            preds.append(1)
        else:
            preds.append(0)

# Calc accuracy
correct = 0
for label, pred in zip(labels, preds):
    if label == pred:
        correct += 1

accuracy_highConf = correct/len(labels)

# Save final results
with open(args.results_dir + "/HighestConf.txt", 'w') as file:
    file.write("label\tpredict\turl_confidence\thtml_confidence\ttab_confidence\tAccuracy: " + str(accuracy_highConf) + "\n")
    for label, pred, url, html, tab in zip(labels, preds, conf_urls, conf_htmls, conf_tabs):
        file.write(f"{label}\t{pred}\t{url}\t{html}\t{tab}\n")

# ---------------------------------------------Vote stacking function---------------------------------------------
'''
- For every entry:
    - Adds up the prediction labels from each model (labels are either 1 or -1)
    - If neg then benign and if pos then mal
    - If no HTML model results exist we can have a tie vote.
        - The model with the highest confidence wins the tie breaker
'''

conf_urls = []
conf_htmls = []
conf_tabs = []
preds = []

for url, html, tab in zip(url_results, html_results, tab_results):
    conf_urls.append(url[1])
    conf_htmls.append(html[1])
    conf_tabs.append(tab[1])

    # Convert tab prediction labels from 0 to -1
    if tab[1] == 0:
        tab[1] = -1

    # Handle HTML None case
    if html[1] != None:
        vote = url[1] + html[1] + tab[1]
    else: 
        vote = url[1] + tab[1]
        
    if vote > 0:
        preds.append(1)
    elif vote == 0:
        highest = max(abs(url[2]), abs(tab[2]))
        
        if highest == abs(url[2]):
            if url[1] > 0:
                preds.append(1)
            else:
                preds.append(0)
        elif highest == abs(tab[2]):
            if tab[1] > 0:
                preds.append(1)
            else:
                preds.append(0)
    else:
        preds.append(0)

# Calc accuracy
correct = 0
for label, pred in zip(labels, preds):
    if label == pred:
        correct += 1

accuracy_vote = correct/len(labels)

# Save final results
with open(args.results_dir + "/Vote.txt", 'w') as file:
    file.write("label\tpredict\turl\thtml\ttab\tAccuracy: " + str(accuracy_vote) + "\n")
    for label, pred, url, html, tab in zip(labels, preds, conf_urls, conf_htmls, conf_tabs):
        if tab == 0:
            tab = -1

        file.write(f"{label}\t{pred}\t{url}\t{html}\t{tab}\n")

# -----------------------------------------------------------------------------------------------------------------

print("Accuracy Mean: " + str(accuracy_mean))
print("Accuracy Vote: " + str(accuracy_vote))
print("Accuracy Highest Confidence: " + str(accuracy_highConf))

# Appends accuracies of all models and stacking methods to accuracies.txt
url_acc, html_acc, tab_acc = accuracy_get()
with open(args.results_dir + "/accuracies.txt", "a") as file:
    file.write(f"{url_acc:0<5}\t{html_acc}\t{tab_acc}\t{accuracy_mean}\t{accuracy_vote}\t{accuracy_highConf}\n")