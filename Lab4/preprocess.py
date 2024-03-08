import pandas as pd
import numpy as np

from PIL import Image, ImageFile
import cv2

from tqdm import trange

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('/home/penny644/DL/train_img.csv')
        label = pd.read_csv('/home/penny644/DL/train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('/home/penny644/DL/test_img.csv')
        label = pd.read_csv('/home/penny644/DL/test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

def prerpocessing(path, mode, img_name, root):
    # solve truncated image
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    img = cv2.imread(path)
    eyes = np.where(img > 50) 
    # ensure eyes is not empty
    if(len(np.unique(eyes[0]))!=0):
        eyes_rows = [min(np.unique(eyes[0])), max(np.unique(eyes[0]))] 
    else:
        w, h, c = img.shape
        eyes_rows = [0, w]
    if(len(np.unique(eyes[1]))!=0):
        eyes_cols = [min(np.unique(eyes[1])), max(np.unique(eyes[1]))]
    else:
        w, h, c = img.shape
        eyes_cols = [0, h]
    crop_img = img[eyes_rows[0]:eyes_rows[1], eyes_cols[0]:eyes_cols[1],:]
    
    # ensure crop_img is not empty
    w, h, c = crop_img.shape
    if(w == 0) & (h == 0):
        crop_img = img
    elif(w == 0):
        eyes_rows = [0, w]
        crop_img = img[eyes_rows[0]:eyes_rows[1], eyes_cols[0]:eyes_cols[1],:]
    elif(h == 0):
        eyes_cols = [0, h]
        crop_img = img[eyes_rows[0]:eyes_rows[1], eyes_cols[0]:eyes_cols[1],:]
    
    # find the size of crop_img
    w, h, c = crop_img.shape
    # make the image to square image
    size = max(w, h)
    new_img = np.zeros((size,size,c), dtype=img.dtype)
    for i in range(c):
        new_img[:,:,i] = crop_img[0,0,i]
    new_img[int((size-w)/2):int((size-w)/2+w), int((size-h)/2):int((size-h)/2+h),:] = crop_img
    new_img = cv2.resize(new_img, (512, 512))
    if mode == 'train':
        path = str(root) + 'preprocessed_train/' + img_name + '.jpeg'
    else:
        path = str(root) + 'preprocessed_test/' + img_name + '.jpeg'
    cv2.imwrite(path, new_img)
    return False

if __name__ == "__main__":
    
    img_name, label = getData('train')
    for i in trange(len(img_name)):
        path = '/home/penny644/DL/new_train/' + img_name[i] + '.jpeg'
        prerpocessing(path, 'train', img_name[i], '/home/penny644/DL/')
    
    img_name, label = getData('test')
    for i in trange(len(img_name)):
        path = '/home/penny644/DL/new_test/' + img_name[i] + '.jpeg'
        prerpocessing(path, 'test', img_name[i], '/home/penny644/DL/')