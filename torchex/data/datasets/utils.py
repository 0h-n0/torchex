
import requests

def download(url, root_path):
    requests.urlretrieve(url, root_path)
    pass
