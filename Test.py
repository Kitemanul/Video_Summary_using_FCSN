import cv2
import numpy as np
import h5py
from tqdm import tqdm
import json

data = {
    'name' : 'myname',
    'age' : 100,
}

with open('test.json', 'w') as f:
    json.dump(data, f)

