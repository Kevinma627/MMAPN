import argparse
from utils import read_data, get_html

parser = argparse.ArgumentParser(description="Fetch HTML Content From URLs")
parser.add_argument('--URL_file', type=str, default='Test_Data/test_all.txt', help="URL filepath")
parser.add_argument('--save_file', type=str, default='Test_Data/test_html_all.txt', help="File to save domain data")
args = parser.parse_args()

urls, labels = read_data(args.URL_file)
    
htmls = []
count = 0
success = 0
for url, label in zip(urls, labels):
    htmls.append(get_html([url, label]))
    count += 1
    
    if(htmls[-1][1] != None and htmls[-1][1][0] == '<'):
        success += 1

    print(count)
    print(success)

    # Every 500 URLs append HTML content to the file
    if(count % 500 == 0 and count != 0):
        with open(args.save_file, 'a', newline='', encoding='utf-8') as train_out:
            for label, html in htmls:
                if html != None and html[0] == '<':
                    train_out.write(f"{label}\t{html}\n")
                else:
                    train_out.write(f"{label}\t{None}\n") # If no HTML content was fetched or something unexpected was fetched write None instead
                
        htmls = [] # Clear htmls

# Handle the last batch of URLs
with open(args.save_file, 'a', newline='', encoding='utf-8') as train_out:
    for label, html in htmls:
        if html != None and html[0] == '<':
            train_out.write(f"{label}\t{html}\n")
        else:
            train_out.write(f"{label}\t{None}\n")