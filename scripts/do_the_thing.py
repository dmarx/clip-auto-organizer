# this is the script that does the thing

#... something like this

#!pip install imgbeddings

import requests
from PIL import Image
from pathlib import Path 

from imgbeddings import imgbeddings
ibed = imgbeddings()

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
def get_img_paths(root, ext='png'):
    return list( Path(root).glob(f'*.{ext}'))
    
def open_image(fpath):
    return Image.open(requests.get(fpath, stream=True).raw)

def featurize_image(image):
    return ibed.to_embeddings(image)

images = get_img_paths(root='~/downloads')
embeddings = [featurize_image(open_image(image)) for image in images]
                
# ...
