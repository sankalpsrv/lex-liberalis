'''This module utilises the IndianKanoon API
You must register for your own key - https://api.indiankanoon.org'''

import requests
from collections import defaultdict
import remover
from bs4 import BeautifulSoup
import inspect
import textwrap
import sys
import os
import re

token_api = api_key = os.getenv('IK_API_KEY')

# Check if the API key is present

# IndianKanoon API - https://api.indiankanoon.org
if not api_key:
    raise ValueError("API key not found. Make sure to set the IK_API_KEY environment variable.")

headers = {
            'authorization': f"Token {token_api}"
        }


def get_titles(searchquery):
    global headers
    url = f'https://api.indiankanoon.org/search/?formInput={searchquery}&pagenum=0'
    documentlist = defaultdict(list)
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()  # Checks for HTTP request errors

        res = response.json()
        for doc in res.get('docs', []):
            doctitle = doc.get('title', 'Unknown Title')
            docid = doc.get('tid', 'Unknown ID')
            documentlist[doctitle].append(docid)
        return documentlist
    
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)
    
    return []  # Return an empty list if there was an error

def get_documents_from_list(searchdocs):
    global headers

    documenttexts = []

    listofids = []
    # Define a regular expression pattern to extract numbers after "doc" and before the next slash
    pattern = r"/doc/(\d+)/"

    url_list = searchdocs.split()

    # Iterate over each URL and extract the document number
    for url in url_list[:10]:  # Considering up to 10 URLs
         # Use re.findall to find all matches in the URL
        matches = re.findall(pattern, url)

        # Check if matches are found
        if matches:
            # Extracted document numbers from the matches
            doc_numbers = ", ".join(matches)
            print(f"URL: {url}, Document Numbers: {doc_numbers}")
            listofids.append(doc_numbers)
        else:
            print(f"URL: {url}, No match found.")
    
    documenttexts = get_documents(listofids)
        

    return documenttexts
        
        

def get_documents(selected_documents):
    global headers
    documenttexts = []
    for idd in selected_documents:
        url = f'https://api.indiankanoon.org/doc/{idd}/'
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()  # Checks for HTTP request errors
            
            res = response.json()
            documenttext = res['doc']
            html_string = str(documenttext)
            escaped_string = bytes(html_string, 'utf-8').decode('unicode-escape')
            soup = BeautifulSoup(escaped_string, "html.parser")
            text = str(soup.get_text())
            initialremove = remover.remove_text(text, start_constant="{'tid", end_constant="'doc': '")
            secondremove = remover.remove_text(initialremove, start_constant="'numcites': ", end_constant="'courtcopy': ")
            documenttexts.append(secondremove)
            
            
        except requests.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            print ("OOps: Something Else",err)
    
    return documenttexts


#if __name__ == "__main__":
    #print(get_titles(searchquery="lgbtq rights"))

    #print(get_documents_from_list(searchdocs="indiankanoon.org/doc/1741837/"))
