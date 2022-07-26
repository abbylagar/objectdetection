# -*- coding: utf-8 -*-

import requests
import tarfile
import zipfile


def download_dataset():
    #dataset
    url = 'https://github.com/abbylagar/drinksdetection/releases/download/drinks_dataset_model/drinks.tar.gz'
    
    #output file
    target_path = 'drinks.tar.gz'

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())

    target_path = 'drinks.tar.gz'
    if target_path.endswith("tar.gz"):
        tar = tarfile.open(target_path, "r:gz")
        tar.extractall(path = './datasets/python/')
        tar.close()
    elif target_path.endswith("tar"):
        tar = tarfile.open(target_path, "r:")
        tar.extractall()
        tar.close()
    print('Done extraction!')


def fetch_zip_file(URL,fname):
    try:
        response = requests.get(URL)
    except OSError:
        print('No connection to the server!')
        return None

    # check if the request is succesful
    if response.status_code == 200:
        # Save dataset to file
        print('Status 200, OK')
        open(fname, 'wb').write(response.content)
    else:
        print('ZIP file request not successful!.')
        return None


def unzipfile():
   # Try to get the weights ZIP file
    URL = 'https://github.com/abbylagar/drinksdetection/releases/download/drinks_dataset_model/mymodel.zip'
    #output file
    output = 'mymodel.zip'
    fetch_zip_file(URL,output)
    
    path_to_zip_file ="./" + output
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall("./weights/")
    print('done extracting weights')    
        

download_dataset()
unzipfile()
