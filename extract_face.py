import numpy as np
import cv2
import matplotlib.pyplot as plt
import random 
from bbox import BBox

IMAGE_SIZE = 64

def read_data_from_txt(path, image_type):
    import os
    result = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            image_name = os.path.join(path, file)
            points = [0] * 136
            """xmin xmax ymin ymax"""   
            result.append([image_name, BBox([230,790,300,820]), points])
    return result

def extract_box(data, output, output_dir, is_training=False, randomize_name=True):
    lines = []
    for (imgPath, bbx, landmarks) in data:
        im = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        imgName = imgPath.split('/')[-1][:-4]
        bbx_sc = bbx
        im_sc = im[max(int(bbx_sc.y), 0):min(int(bbx_sc.y+bbx_sc.h), im.shape[0]), max(int(bbx_sc.x), 0): min(int(bbx_sc.x+bbx_sc.w), im.shape[1]) ]
        im_sc = cv2.resize(im_sc, (IMAGE_SIZE, IMAGE_SIZE))
        name = output_dir+'/'+imgName
        if randomize_name: 
            name += '_{}'.format(random.randint(1, 10000))
        name += 'sc.png'
        cv2.imwrite(name, im_sc)
        lines.append(name + ' ' + ' '.join(map(str, landmarks)) + '\n')
    with open(output, 'w') as fid:
        fid.writelines(lines)

if __name__ == '__main__':
    data = read_data_from_txt("videos/", image_type = ".png")
    extract_box(data, output='video_new.txt', output_dir='augmented_videos', is_training=False)

