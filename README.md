# MMAPN
Multi-Model Anti-Phishing Network: Hybrid system that utilizes three different deep learning models to classify websites as either benign or malicious (phishing). This project idea was derived from the paper *The applicability of a hybrid framework for automated phishing detection*.

## Models

- **URLNet**
    - Input Data: URL
    - Implementation of the URLNet model proposed by the paper *URLNet - Learning a URL Representation with Deep Learning for Malicious URL Detection*.

- **HTMLNet**
    - Input Data: HTML Content
    - Modified implementation of URLNet that takes in HTML content as data instead of URLs. 
    - Inspired by the HTMLPhish model proposed by the paper *HTMLPhish: Enabling Phishing Web Page Detection by Applying Deep Learning Techniques on HTML Analysis*.

- **TabNet**
    - Input Data: Domain/Hosting Information (See the "Data" section for more details)
    - Implementation of TabNet model proposed by the paper *TabNet : Attentive Interpretable Tabular Learning* by DreamQuark.

## Packages and Dependencies 

This project uses many packages and dependencies and requires some to be of a specific version to work correctly. Two of the models (URLNet and HTMLNet) use Tensorflow and the third one (TabNet) uses PyTorch. These two packages share some dependencies but they require them to be different versions. As a result, it is recommended to set up a seperate environment for one of the packages. 

The run.sh script is set up to automatically activate and deactivate a miniconda environment named "torch" so that it can run the TabNet model.

Note the following lines:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
conda deactivate
```

Feel free to modify these lines to match how your environments are set up.

The following are some packages you made need to install (versions listed are known to work):

### Tensorflow
Python 3.10.12
- tensorflow 2.19.0
- pandas 1.5.3

### PyTorch
Python 3.11.11
- torch 1.13.1
- pytorch_tabnet 4.1.0
- pandas  2.2.3

### General
- scikit-learn 1.6.1
- tldextract 5.2.0
- beautifulsoup4 4.13.3
- tqdm 4.67.1

## Data

The Train_Data and Test_Data directories contain sample train and test data for all 3 models for your convenience.
- Test data for all 3 models are taken from the first 1000 URLs from test.txt and match each other one to one.
- train_url.txt is the first 10,000 entries of train.txt

### URLNet/HTMLNet

In all datasets for training or testing, each line includes the label and the
text string following the template:

`<label><tab><string>`

**Note:** For HTML the string can be "None" in the case that HTML content failed to fetch from the URL.
(See the HTML_Fetcher portion of the Other section for more details).

**Example:**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+1  http://www.exampledomain.com/urlpath/...

-1  http://www.exampledomain.com/urlpath/...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

or

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+1  <!DOCTYPE html>
    ...
    </html>

-1  <!DOCTYPE html>
    ...
    </html>

-1  None
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The label is either +1 (Malicious) or -1 (Benign).

### TabNet

Train and test datasets are in CSV format and include the following (in order):
- Features:
    1. Duration
        - Time (in years) between creation and expiration dates.
    2. Domain Age
        - Time since creation date (in years).
    3. Number of Updates
        - Number of times the domain has been updated.
    4. Most Recent Update Time
        - Time since the domain was last updated (in years).
    5. DNSSEC Signature
        - 0 if dnssec is 'unsigned', 1 if dnssec is 'signed delegation', and 2 if dnssec is 'signed'.
    6. Registrar
        - One of 5 common registrars among the domains found in the URL data.
        - Each registrar is keyed with an integer value 1-5.
        - If the registrar does not match any of the 5 returns 0 instead.
    7. Number of Name Servers
    8. Pagerank Score
        - Decimal value from 0-10 that determines a websites pagerank (higher is better).
    9. Pagerank
        - A website's actual pagerank (integer value).
- Label (Either a 0 (benign) or 1 malicious)

**NOTE:** Any missing or invalid data is substituted with a -1.

Examples:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
27.994524298425734,27.71252566735113,2,0.1834360027378508,0,0,4,3.86,1366365,1
22.001368925393567,21.87542778918549,1,0.9089664613278576,0,0,4,3.05,3954719,0
-1.0,13.221081451060916,-1,-1.0,-1,0,1,2.6,8070801,0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first 7 features are fetched through Whois API and the last 2 features are fetched through Open Page Rank API.

## Training

### URLNet

Use the following command to train the URLNet model:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 train_url.py --data_data_dir <train_data_path> --data_dev_pct 0.2 \
--train_nb_epochs <number_of_epochs> --train_batch_size <urls_per_batch> \
--log_output_dir <model_save_location>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example using defaults:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 train_url.py --data_data_dir Train_Data/train_url.txt --data_dev_pct 0.2 \
--train_nb_epochs 2 --train_batch_size 128 \
--log_output_dir Model/runs_url
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After training the following will be saved in the location specified by the log_output_dir parameter:
- A checkpoint of every training epoch
- A character dictionary
- A word dictionary
- A sub-word dictionary

For more training parameter options, a full list of training parameters is listed at the top of train_url.py

### HTMLNet

Use the following command to train the HTMLNet model:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 train_html.py --data_data_dir <train_data_path> --data_dev_pct 0.2 \
--train_nb_epochs <number_of_epochs> --train_batch_size <htmls_per_batch> \
--log_output_dir <model_save_location>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example using defaults:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 train_html.py --data_data_dir Train_Data/train_html.txt --data_dev_pct 0.2 \
--train_nb_epochs 3 --train_batch_size 20 \
--log_output_dir Model/runs_html
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After training the following will be saved in the location specified by the log_output_dir parameter:
- A checkpoint of every training epoch
- A character dictionary
- A word dictionary
- A sub-word dictionary

For more training parameter options, a full list of training parameters is listed at the top of train_html.py

### TabNet

Use the following command to train the TabNet model:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 Tab.py --train_data_file <train_data_path> --data_dev_pct 0.1 --model_save_dir <model_save_location> --model_name <name_of_the_model>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example using defaults:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 Tab.py --train_data_file Train_Data/train_tab.csv --data_dev_pct 0.1 --model_save_dir Model/runs_tab --model_name MMAPN_tab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default the model uses the "balanced accuracy" metric for early stopping.

After training the following will be saved in the location specified by the model_save_dir parameter:
- A zip file containing the saved model (does not need to be unzipped).
- A file named feature_importance.txt that contains information on the importance of each feature during training.
    - Decimal value (higher means more important).
    - Features are indexed in the order listed in the TabNet portion of the Data section.

## Testing
### URLNet

Use the following command to test the URLNet model:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 test_url.py --data_data_dir <test_data_path> \
--log_checkpoint_dir <model_save_location> --log_output_dir <test_result_save_location> \
--test_batch_size <urls_per_batch>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example using defaults:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 test_url.py --data_data_dir Test_Data/test_url.txt \
--log_checkpoint_dir Model/runs_url --log_output_dir Model/runs_url \
--test_batch_size 128
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After testing the following will be saved in the location specified by the log_output_dir parameter:
- A file named test_results.txt
    - Contains the true label, model prediction, and model score for every test entry (-1 is benign and 1 is malicious).
    - The test accuracy can be found at the top of the file.

**IMPORTANT:** You might need to adjust the prediction threshold on line 115 of test_url.py (the float value) if you're testing a freshly trained model.
- Choose one that best divides the prediction results into the correct classes.
- Make sure to change the URL_THRESH constant in stack.py to match (see the Model Stacking section for more details).

The default threshold is 0.72:

```python
predictions = (logits.numpy() > 0.72).astype(int).flatten()
```

For more testing parameter options, a full list of testing parameters is listed at the top of test_html.py

Ensure that the relevant parameters are conistent across training and testing.

### HTMLNet

Use the following command to test the HTMLNet model:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 test_html.py --data_data_dir <test_data_path> \
--log_checkpoint_dir <model_save_location> --log_output_dir <test_result_save_location> \
--test_batch_size <htmls_per_batch>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example using defaults:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 test_html.py --data_data_dir Test_Data/test_html.txt \
--log_checkpoint_dir Model/runs_html --log_output_dir Model/runs_html \
--test_batch_size 20
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After testing the following will be saved in the location specified by the log_output_dir parameter:
- A file named test_results.txt
    - Contains the true label, model prediction, and model score for every test entry (-1 is benign and 1 is malicious).
    - The test accuracy can be found at the top of the file.

**IMPORTANT:** You might need to adjust the prediction threshold on line 116 of test_url.py (the float value) if you're testing a freshly trained model.
- Choose one that best divides the prediction results into the correct classes.
- Make sure to change the HTML_THRESH constant in stack.py to match (see the Model Stacking section for more details).

The default threshold is 0.54:

```python
predictions = (logits.numpy() > 0.54).astype(int).flatten()
```

**Note:** The model does not make predictions on entries where the HTML content is "None" and the accuracy does not take these entries into account (See the HTML_Fetcher portion of the Other section for more details).

For more testing parameter options, a full list of testing parameters is listed at the top of test_html.py

Ensure that the relevant parameters are conistent across training and testing.

### TabNet

Use the following command to test the TabNet model:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 test_tab.py --test_data_file <test_data_path> --model_save_dir <model_save_location> --model_name <name_of_the_model>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example using defaults:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 test_tab.py --test_data_file Test_Data/test_tab.csv --model_save_dir Model/runs_tab --model_name MMAPN_tab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After testing the following will be saved in the location specified by the model_save_dir parameter:
- A file named test_results.txt
    - Contains the true label, model prediction, and prediction confidence for every test entry (0 is benign and 1 is malicious).
    - The test accuracy can be found at the top of the file.
    - Information on the importance of each feature during testing can be found at the bottom of the file.
        - Decimal value (higher means more important).
        - Features are indexed in the order listed in the TabNet portion of the Data section.

Ensure that the relevant parameters are conistent across training and testing.

## Model Stacking

MMAPN combines the results from each of its 3 models to make a final predicition. It has 3 different stacking functions:
- Weighted-Mean:
    - Confidences are calculated for all 3 model predictions and are normalized.
    - The confidences are signed based on the predicted class and then added together.
    - The final prediction is made based on if this score is postive or negative.
    - If there are no prediction results from the HTML model for a particular entry that model is ignored. (HTML content None case - See the HTML_Fetcher portion of the Other section for more details).
- Highest Confidence:
    - Confidences are calculated in the same way as the Weighted-Mean function and are normalized.
    - The final prediction is the prediction result of whichever model has the highest confidence.
    - If there are no prediction results from the HTML model for a particular entry that model is ignored.
- Vote:
    - The model predictions are taken as a "vote" for the respective class.
    - Whichver class has the majority vote is the final prediction.
    - If there are no prediction results from the HTML model for a particular entry that model is ignored.
        - In this case the vote can now tie. If this happens, the model with the highest confidence wins out.

For more detailed explanations of how the final predictions are calculated for each stacking function read the comments within the stack.py script.

Running the following command will automatically apply the stacking functions to test results:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 stack.py --url_save_dir <url_model_save_location> --html_save_dir <html_model_save_location> --tab_save_dir <TabNet_model_save_location> --results_dir <results_save_location>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After running stack.py the following will be saved in the location specified by the results_dir parameter:
- WeightedMean.txt
    - Contains the true label, prediction result, prediction score, and confidences of each model.
    - The accuracy can be found at the top of the file.
- HighestConf.txt
    - Contains the true label, prediction result, and confidences of each model.
    - The accuracy can be found at the top of the file.
- Vote.txt
    - Contains the true label, prediction result, and predictions of each model.
    - The accuracy can be found at the top of the file.
- accuracies.txt
    - Everytime stack.py is ran it will append the accuracies of all 3 models and all 3 stacking functions to the accuracies.txt file

**IMPORTANT:** Make sure the URL_THRESH and HTML_THRESH constants at the top of stack.py match the thresholds in test_url.py and test_html.py respectively (See the Testing section for more details).

Default threshold values:

```python
URL_THRESH = 0.72
HTML_THRESH = 0.54
```

### Automatically Test Everything

You can automatically test all 3 models and then run stack.py on the results by running run.sh

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
./run.sh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script calls the test script for all 3 models and stack.py with all default parameters (including filepaths and directories). You are welcome to change commands in the script to use custom parameters if needed.

Remember that stack.py will still append accuracies to accuracies.txt

## Other

### HTML_Fetcher

This script is used to fetch HTML content for all the entries in a URL data file. (Make sure the file follows the format described in the Data section).

It can be ran with the following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 HTML_Fetcher.py --URL_file <url_data_file> --save_file <html_data_file>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- It will append HTML data in batches of 500 entries (in the format described in the Data section) to the file specified by the save_file parameter.
    - Any content already in the file WILL NOT be overwritten.
- It make take a while to process large URL data files.
- It may not successfully fetch HTML content for every URL. In this case it will simply write "None" where the HTML content would be.
- For every URL that is processed, two integers are printed to the terminal. The first is the total number of URLs processed. The second is the number of HTMLs fetched successfully.

### get_tab

This script is used to fetch domain/hosting data for all entries in a URL data file. (Make sure the file follows the format described in the Data section).

It can be ran with the following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 HTML_Fetcher.py --URL_file <url_data_file> --save_file <tab_data_csv>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- It will save domain/hosting data to the CSV file specified by the save_file parameter. (See the TabNet portion of the Data section for information on features).
    - **WARNING:** This WILL overwrite any content already in the CSV file.
- It make take a while to process large URL data files.
- Any missing or invalid data is substituted by a -1.
- For every URL that is processed the number of processed URLs will print to the terminal.
- Data is fetched through the following APIs:
    - Whois API
    - Open Page Rank API

### Sample Train/Test Runs

The MMAPN.ipynb jupyter notebook file features a sample train and sample test run for all 3 models. Here you can see what the terminal output of these runs might look like.

## References

### Papers
- van Geest, R. J., Cascavilla, G., Hulstijn, J., & Zannone, N. (2024). The applicability of a hybrid framework for automated phishing detection. Computers & Security, 139, 103736.    doi:10.1016/j.cose.2024.103736
- Hung Le, Quang Pham, Doyen Sahoo, & Steven C. H. Hoi. (2018). URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection.
- Opara, C., Wei, B., & Chen, Y. (2020). HTMLPhish: Enabling Phishing Web Page Detection by Applying Deep Learning Techniques on HTML Analysis. In 2020 International Joint Conference on Neural Networks (IJCNN) (pp. 1–8). IEEE.
- Sercan O. Arik, & Tomas Pfister. (2020). TabNet: Attentive Interpretable Tabular Learning.

### Code

- The code related to the URLNet model including train_url.py, test_url.py, and many of the functions in utils.py was built off code taken from the following GitHub repository: [https://github.com/ggkunka/URLNet](https://github.com/ggkunka/URLNet).
    - It is a fork of the URLNet GitHub repository ([https://github.com/Antimalweb/URLNet](https://github.com/Antimalweb/URLNet)).
    - The code related to the HTMLNet model including train_html.py and test_html.py also uses this code as a guideline.
- PyTorch TabNet by DreamQuark was the implementation used for the TabNet model ([https://github.com/dreamquark-ai/tabnet](https://github.com/dreamquark-ai/tabnet)).

### APIs

- Whois API ([https://rapidapi.com/HemKrishLabs/api/whois-api6](https://rapidapi.com/HemKrishLabs/api/whois-api6))
- Open Page Rank ([https://www.domcop.com/openpagerank/](https://www.domcop.com/openpagerank/))