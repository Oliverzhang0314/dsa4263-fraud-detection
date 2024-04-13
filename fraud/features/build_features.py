import os
import re
import tldextract
from urllib.parse import urlparse, parse_qs, unquote
import dns.resolver
from googlesearch import search
from IPython.display import display

def process_new_url(df, path="../data/processed"):
    '''
    parse 'url' column of df into domain, directory, file and params.
    Also providing length of each property.
    Adding 10 new columns, saving one local csv: 'result.csv'
    '''
    # Remove trailing space and quotation
    df['parsed_url'] = df['url'].apply(lambda x: remove_trailing(x))
    
    # Remove scheme
    df['parsed_url'] = df['parsed_url'].apply(lambda x: remove_scheme(x))
    
    # Decode
    df['parsed_url'] = df['parsed_url'].apply(lambda x: decode(x))
    
    df['domain'] = df['parsed_url'].apply(lambda x: parse_domain(x))

    # Extract directory
    df['directory'] = df['parsed_url'].apply(lambda x: parse_dir(x))

    # Extract file
    df['file'] = df['parsed_url'].apply(lambda x: parse_file(x))
    
    # Extract params
    df['params'] = df['parsed_url'].apply(lambda x: parse_params(x))
    
    # Add length-related columns
    df.loc[:, 'length_url'] = df['parsed_url'].str.len()
    df.loc[:, 'domain_length'] = df['domain'].str.len()
    df.loc[:, 'directory_length'] = df['directory'].str.len()
    df.loc[:, 'file_length'] = df['file'].str.len()
    df.loc[:, 'params_length'] = df['params'].str.len()
        
    # Upload dataset 
    filename="malicious-urls-dataset/result.csv"
    file_path = os.path.join(path, filename)
    df.to_csv(file_path, index=False)
    
    return df

def remove_trailing(url):
    return url.strip().strip("'").strip()

def remove_scheme(url):
    parsed = urlparse(url)
    scheme = "%s://" % parsed.scheme
    return parsed.geturl().replace(scheme, '', 1)

def decode(url):
    return unquote(url)

def parse_domain(url):
    try:
        return urlparse("https://" + urlparse(url).geturl()).netloc
    except:
        return urlparse(url).netloc
    
def parse_dir(url):
    file_ext = [".htm", ".html", ".gif", ".jpg", ".png", ".js", ".java", ".class", ".txt", 
                ".php", ".php3", ".shtm", ".shtml", ".asp", ".cfm", ".cfml", ".cgi", ".do", ".ars"]
    
    url_split = url.strip("/").split("/")
    if len(url_split) <= 1:
        return ""
    elif len(url_split) == 2: # check if last chunk is file or dir
        last_chunk = url_split[-1]
        
        # last chunk is file
        for ext in file_ext:
            if last_chunk.find(ext) != -1: # file extension is found
                return ""
        
        # last chunk is dir
        if "?" in last_chunk: # with params
            return last_chunk.split("?")[0]
        else: # w/o params
            return last_chunk
    else: # check if last chunk is file or dir
        last_chunk = url_split[-1]
        
        # last chunk is file
        for ext in file_ext:
            if last_chunk.find(ext) != -1: # file extension is found
                return "/".join(url_split[1:-1])
            
        # last chunk is dir
        if "?" in last_chunk: # with params
            return "/".join(url_split[1:-1]) + last_chunk.split("?")[0] 
        else: # w/o params
            return "/".join(url_split[1:-1]) + "/" + last_chunk
    
def parse_file(url):
    file_ext = [".htm", ".html", ".gif", ".jpg", ".png", ".js", ".java", ".class", ".txt", 
                ".php", ".php3", ".shtm", ".shtml", ".asp", ".cfm", ".cfml", ".cgi", ".do", ".ars"]
    
    url_split = url.split("/")
    if len(url_split) <= 1:
        return ""
    else:
        last_chunk = url_split[-1]
        for ext in file_ext:
            if last_chunk.find(ext) != -1: # file extension is found
                if "?" in last_chunk: # with params
                    return last_chunk.split("?")[0]
                else: # w/o params
                    return last_chunk
        return "" # file extension not found

def parse_params(url):
    return urlparse(url).query

def special_chars_qty(df):
    vowels=['a','e','i','o','u']
    features = {'at':'@', 'questionmark':'?', 'underline':'_', 'hyphen':'-', 'equal':'=', 'dot':'.', 
            'hashtag':'#', 'percent':'%', 'plus':'+', 'dollar':'$', 'exclamation':'!', 'asterisk':'*', 
            'comma':',', 'slash':'/', 'space':' ', 'tilde':'~','and':'&'}
    cols=['url','domain','directory','file','params']

    # add quantity of special characters for all cols
    for i in range(len(cols)):
        for key, value in features.items():
                df.loc[:, "qty_" + key + '_'+ cols[i]] = df.loc[:, cols[i]].apply(lambda x: x.count(value) if x else 0)

    # add vowel qty for domain
    df.loc[:,"qty_vowels_domain"] = df.loc[:,'domain'].apply(lambda x: 0 if x is None else sum(char.lower() in vowels for char in x))

    return df   

def shortened(url):
    match = re.search(
                      'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0 

# check if existed in google
def check_google_index(url):
    site = search(url, 10)
    return 1 if site else 0

# check if email in url
def check_email(url):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    if(re.fullmatch(regex, url)):
        return 1
    else:
        return 0

# check if domain has spf record
def check_spf(domain_name):
    """
    Get the SPF record recorded in DNS for a specific domain name. Returns None if not found.
    """
    try:
        spf_records = dns.resolver.query(domain_name) 
        return 1
    except:
        return 0

# check the quantity of tlds in url
def count_tlds(url):
    # Regular expression pattern to extract TLDs from a URL
    tld_pattern = r'\.([a-zA-Z]{2,})$'
    
    # Find all matches of the TLD pattern in the URL
    tlds = re.findall(tld_pattern, url)
    
    # Return the count of unique TLDs
    return len(set(tlds))

# check the number of params in url
def count_params(url):
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract the query parameters
    query_params = parsed_url.query
    
    # Parse the query parameters
    parsed_query_params = parse_qs(query_params)
    
    # Count the number of parameters
    num_params = len(parsed_query_params)
    
    return num_params

# use this to complete preprocessing of new dataset 
def reformat_df(df):
    '''
    modify df inplace, adding in extracted features
    '''
     
    df_new = special_chars_qty(df)
    df_new.loc[:,'url_shortened'] = df_new.loc[:,'url'].apply(lambda x: shortened(x))

    # add the quantity of params in url
    df_new.loc[:,'qty_params'] = df_new.loc[:,'url'].apply(lambda x: count_params(x))

    # check if google index is available for url & domain
    df_new.loc[:,'url_google_index'] = df_new.loc[:,'url'].apply(lambda i: check_google_index(i))
    df_new.loc[:,'domain_google_index'] = df_new.loc[:,'domain'].apply(lambda i: check_google_index(i)) 

    # check if email is in utl
    df_new.loc[:,'email_in_url'] = df_new.loc[:,'url'].apply(lambda i: check_email(i))

    # check qty of tld in url
    df_new.loc[:,'qty_tld_url']=df_new.loc[:,'url'].apply(count_tlds)
    
    # Check if TLD is present in params column and return 1 or 0
    df_new['tld_present_params'] = df_new.apply(lambda row: 1 if tldextract.extract(row['params']).suffix in row['domain'] else 0, axis=1)

    # Create a mapping dictionary
    label_mapping = {'benign': 0, 'phishing': 1}
    df_new['phishing'] = df_new['type'].map(label_mapping)
    
    # Drop extra columns
    df_new = df_new.reset_index(drop=True)
    display(df_new.head(5))
    display(df.describe())
    df_new = df_new.drop(columns=['url','domain','directory','params','file', 'type', 'parsed_url'])

    return df_new

#### Count redirects
# def count_redirects(url):
#     try:
#         response = requests.get(url, allow_redirects=True)
#         num_redirects = len(response.history)
#         return num_redirects
#     except requests.exceptions.RequestException:
#         # If an exception occurs during the request (e.g., invalid URL), return -1
#         return -1

#### Count TTL for hostname of url
# def get_ttl(url):
#     try:
#         result = dns.resolver.resolve(url, 'A')
#         if result.response.answer:
#             return int(result.response.answer[0].ttl)
#     except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN,
#             dns.exception.Timeout, dns.resolver.NoNameservers):
#         return 0
#     return 0

#### Check SSL Cert
# import requests
# def check_ssl_certificate(url):
#     try:
#         response = requests.head(url)
#         # Check if the response status code is 200 (OK) or 301 (Moved Permanently)
#         if response.status_code == 200 or response.status_code == 301:
#             return 1  # SSL certificate is available
#     except requests.exceptions.SSLError:
#         pass  # SSL certificate is not available or there is an SSL error
#     except requests.exceptions.RequestException:
#         pass  # Handle other request exceptions if needed
#     return 0  # SSL certificate is not available