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
        if file.endswith(".pts"):
            image_name = os.path.join(path, file)
            with open(image_name, 'r') as fid:            
               components = fid.read().split("\n")
               points = []
               for p in components[3:-1]:
                  point = list(map(float, p.strip().split(" ")))
                  points.append(point)
               x_min = int(min([a[0] for a in points]))
               x_max = int(max([a[0] for a in points]))
               y_min = int(min([a[1] for a in points]))
               y_max = int(max([a[1] for a in points]))
               x_margin = (x_max - x_min) * 0.1
               y_margin = (y_max - y_min) * 0.1
               result.append([image_name.replace(".pts", image_type), BBox([x_min-x_margin, x_max+x_margin, y_min-y_margin, y_max+y_margin]), 
                     points])
    return result

def flip(face, landmark):
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return (face_flipped_by_x, landmark_)

def rotate(img, bbox, landmark, alpha):
    center = (bbox.x+bbox.w/2, bbox.y+bbox.h/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, img.shape)
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[bbox.y:bbox.y+bbox.h,bbox.x:bbox.x+bbox.w]
    return (face, landmark_)

def processImage(imgs):
    """
        process images before feeding to CNNs
        imgs: N x 1 x W x H
    """
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        m = img.mean()
        s = img.std()
        imgs[i] = (img - m) / s
    return imgs

def data_augmentation(data, output, output_dir, is_training=False, randomize_name=True):
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
        lm_sc = bbx_sc.normalizeLmToBbx(landmarks)
        lines.append(name + ' ' + ' '.join(map(str, lm_sc.flatten())) + '\n')
       
        if not is_training:
            continue

        try:
        
           flipsc, lm_flipsc = flip(im_sc, lm_sc)
           name = output_dir+'/'+output_dir+'_'+imgName+'flipsc.png'
           cv2.imwrite(name, flipsc)
           lines.append(name + ' ' + ' '.join(map(str, lm_flipsc.flatten())) + '\n')
        except:
           print("safe")
           None

    with open(output, 'w') as fid:
        fid.writelines(lines)

if __name__ == '__main__':
    data = read_data_from_txt("datasets/lfpw-trainset/", image_type = ".png")
    data2 = read_data_from_txt("datasets/helen-trainset/", image_type = ".jpg")
    #data3 = read_data_from_txt("datasets/ibug-trainset/", image_type = ".jpg")
    data4 = read_data_from_txt("datasets/afw-trainset/", image_type = ".jpg")
    data5 = read_data_from_txt("datasets/lfpw-testset/", image_type = ".png")
    data6 = read_data_from_txt("datasets/helen-testset/", image_type = ".jpg")
    data_augmentation(data + data2 + data4 + data5 + data6 , output='train_new.txt', output_dir='augmented_train', is_training=False)

    data = read_data_from_txt("datasets/lfpw-testset/", image_type = ".png")
    data2 = read_data_from_txt("datasets/helen-testset/", image_type = ".jpg")
    data_augmentation(data + data2, output='test_new.txt', output_dir='augmented_test', is_training=False)

