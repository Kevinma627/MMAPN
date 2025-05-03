import argparse
import requests
import tldextract
import pandas as pd
from datetime import datetime
from utils import read_data

# Extracts domain from URL
def extract_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"

# Fetches whois data from whois-api
def fetch_whois(domain):
    url = "https://whois-api6.p.rapidapi.com/whois/api/v1/getData"
    payload = {"query": domain}
    headers = {"x-rapidapi-key": "c349d92282msh62ed97dd1d91ae9p1cff6ajsn4d7a0e324335",
	"x-rapidapi-host": "whois-api6.p.rapidapi.com",
	"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    return response.json() if response.status_code == 200 else None

# Fetches page rank data from openpagerank
def fetch_rank(domains):
    url = "https://openpagerank.com/api/v1.0/getPageRank"
    headers = {"API-OPR": "wgo8g8cogok8csg8o8w08sg40cckcc8s40o8kowg"}

    payload = []
    for domain in domains:
        payload.append(('domains[]', domain))

    response = requests.get(url, headers=headers, params=payload)
    return response.json() if response.status_code == 200 else None

def parse_date(date, recent):
    '''
    - Parses dates and returns them in a standard format (Y-M-D H:M:S)
    - If it can't parse a date due to its format (even after cleaning) returns None
    - Can take in a list of dates and will parse the most recent one if recent == True otherwise parses the oldest date
    '''
    if(isinstance(date, str)):
        dt_clean = clean_date(date)
        try:
            return datetime.strptime(dt_clean, "%Y-%m-%d %H:%M:%S")
        except:
            return None
    else:
        dates = []
        for dt in date:
            dt_clean = clean_date(dt)
            dates.append(datetime.strptime(dt_clean, "%Y-%m-%d %H:%M:%S"))

        if(recent):
            return max(dates)
        return min(dates)

# Fixes dates with unexpected formats
def clean_date(date):
    if '/' in date:
        date = date.replace("/", "-")
    
    return date[:19]

# Returns time (in years) between creation and expiration dates. If either date is None returns -1 instead.
def get_duration(creation_date, expiration_date):
    creation = parse_date(creation_date, False)
    expiration = parse_date(expiration_date, False)

    if(creation == None or expiration == None):
        return -1

    duration = (expiration - creation).days / 365.25
    return duration

# Returns time since creation date (in years). If the creation date is None returns -1 instead.
def get_domain_age(creation_date):
    domain_date = parse_date(creation_date, False)
    if(domain_date == None):
        return -1
    return (datetime.now() - domain_date).days / 365.25


def get_update_data(update):
    '''
    - Returns the number of updates from a list of updates.
    - Returns how long ago the last update was (in years). If the time cannot be parsed returns -1.
    '''
    if(isinstance(update, str)): # If update is a str that means there's only one update
        ups = 1
    else:
        ups = len(update)

    time = parse_date(update, True)
    if(time):
        update_time = (datetime.now() - time).days / 365.25
    else:
        update_time = -1
    
    return ups, update_time

def get_registrar_data(registrar):
    '''
    - Matches the registrar with a one of 5 common registrars among the domains found in the URL data.
    - Each registrar is keyed with an integer value 1-5.
    - If the registrar does not match any of the 5 returns 0 instead.
    '''
    match registrar:
        case "GoDaddy.com, LLC":
            return 1
        case "CloudFlare, Inc.":
            return 2
        case "Loopia AB":
            return 3
        case "MarkMonitor, Inc.":
            return 4
        case "NameSilo, LLC":
            return 5
        case _:
            return 0

# Returns 0 if dnssec is 'unsigned', 1 if dnssec is 'signed delegation', and 2 if dnssec is 'signed'
def dnssec_parse(dnssec):
    if("unsigned" in dnssec):
        return(0)
    elif("delegation" in dnssec):
        return(1)
    else:
        return(2)
    
parser = argparse.ArgumentParser(description="Fetch Domain Data From URLS")
parser.add_argument('--URL_file', type=str, default='Test_Data/test_all.txt', help="URL filepath")
parser.add_argument('--save_file', type=str, default='Test_Data/test_tab_all.csv', help="CSV to save domain data")
args = parser.parse_args()

urls, labels = read_data(args.URL_file)
domains = [extract_domain(url) for url in urls]
durations = []
ages = []
updates = []
update_times = []
dnssec = []
registrars = []
nameservers = []
rank_scores = []
ranks = []
batch = []

completed = 0

for domain in domains:
    # Fetches data from openpagerank in batches of 100 (max domains in a query == 100) and appends data to appropriate lists.
    batch.append(domain)

    if(len(batch) == 100):
        rank_dics = fetch_rank(batch)["response"]
        
        for dic in rank_dics:
            if(dic["status_code"] == 200):
                rank_scores.append(dic["page_rank_decimal"])
                ranks.append(dic["rank"])
            else:
                rank_scores.append(-1)
                ranks.append(-1)
        
        batch = []

    # Fetch whois data for each domain
    whois_data = fetch_whois(domain)
    completed += 1

    print(whois_data)
    print(completed)

    '''
    If a valid result is returned from whois-api we parse through the data in the response and append to appropriate lists.
    We append -1 to if any data is missing or invalid.
    If there is no valid result then we append -1 to every list.
    Most data is in the form of a string, however some can sometimes be lists of strings. Such cases are handled accordingly
    here or in the respective helper function.
    '''
    if(whois_data != None and "result" in whois_data):
        whois_data = whois_data["result"]
        
        # Durations and Ages
        if("creation_date" in whois_data and whois_data["creation_date"] != None):
            ages.append(get_domain_age(whois_data["creation_date"]))
        
            if("expiration_date" in whois_data and whois_data["expiration_date"] != None):
                durations.append(get_duration(whois_data["creation_date"], whois_data["expiration_date"]))
            else:
                durations.append(-1)
        else:
            durations.append(-1)
            ages.append(-1)

        # Number of Updates and Last Update Time
        if("updated_date" in whois_data and whois_data["updated_date"] != None):
            ups, update_time = get_update_data(whois_data["updated_date"])
        else:
            ups = -1
            update_time = -1

        updates.append(ups)
        update_times.append(update_time)

        # Registrar
        if("registrar" in whois_data and whois_data["registrar"] != None):
            registrars.append(get_registrar_data(whois_data["registrar"]))
        else:
            registrars.append(0)

        # Number of name servers
        if("name_servers" in whois_data and whois_data["name_servers"] != None):
            if(isinstance(whois_data["name_servers"], str)):
                nameservers.append(1)
            else:
                nameservers.append(len(whois_data["name_servers"]))
        else:
            nameservers.append(0)
        
        # DNSSEC signature
        if("dnssec" in whois_data and whois_data["dnssec"] != None):
            if(isinstance(whois_data["dnssec"], str)):
                dnssec.append(dnssec_parse(whois_data["dnssec"].lower()))
            else:
                for i in range(len(whois_data["dnssec"])):
                    if("delegation" not in whois_data["dnssec"][i].lower()):
                        dnssec.append(dnssec_parse(whois_data["dnssec"][i].lower()))
                        break
                    elif(i == len(whois_data["dnssec"]) - 1):
                        dnssec.append(dnssec_parse(whois_data["dnssec"][0].lower()))
        else:
            dnssec.append(-1)
    else:
        durations.append(-1)
        ages.append(-1)
        updates.append(-1)
        update_times.append(-1)
        dnssec.append(-1)
        registrars.append(-1)
        nameservers.append(-1)

# Handle the last batch of page rank fetching if domains % 100 != 0
if(batch):
    rank_dics = fetch_rank(batch)["response"]
        
    for dic in rank_dics:
        if(dic["status_code"] == 200):
            rank_scores.append(dic["page_rank_decimal"])
            ranks.append(dic["rank"])
        else:
            rank_scores.append(-1)
            ranks.append(-1)

data = {
    "duration": durations,
    "domain_age": ages,
    "updates": updates,
    "update_time": update_times,
    "dnssec": dnssec,
    "registrar": registrars,
    "nameservers": nameservers,
    "rank_score": rank_scores,
    "page_rank": ranks,
    "label": labels
}

df = pd.DataFrame(data)
df.to_csv(args.save_file, mode='w', index=False)