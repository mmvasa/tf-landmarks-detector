import numpy as np
import cv2
import matplotlib.pyplot as plt

AUGMENTED_PATH = 'augmented-data/'
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
               result.append([image_name.replace(".pts", image_type), BBox([x_min-x_margin, x_max+x_margin, y_min-y_margin,y_max+y_margin]), 
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

def data_augmentation(data, output, is_training=False):
    lines = []
    dst = AUGMENTED_PATH
    for (imgPath, bbx, landmarks) in data:
        im = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        imgName = imgPath.split('/')[-1][:-4]
        
        bbx_sc = bbx #.bbxScale(im.shape, scale=1)
        im_sc = im[int(bbx_sc.y):int(bbx_sc.y+bbx_sc.h), int(bbx_sc.x):int(bbx_sc.x+bbx_sc.w)]
        
        try:
           im_sc = cv2.resize(im_sc, (IMAGE_SIZE, IMAGE_SIZE))
           name = dst+imgName+'sc.png'
           cv2.imwrite(name, im_sc)
           lm_sc = bbx_sc.normalizeLmToBbx(landmarks)
           lines.append(name + ' ' + ' '.join(map(str, lm_sc.flatten())) + '\n')
          
           if not is_training:
               continue

           flipsc, lm_flipsc = flip(im_sc, lm_sc)
           name = dst+imgName+'flipsc.png'
           cv2.imwrite(name, flipsc)
           lines.append(name + ' ' + ' '.join(map(str, lm_flipsc.flatten())) + '\n')
        except:
           print("safe")
           None

    with open(output, 'w') as fid:
        fid.writelines(lines)


class BBox(object):

    def __init__(self, bbx):
        self.x = bbx[0]
        self.y = bbx[2]
        self.w = bbx[1] - bbx[0]
        self.h = bbx[3] - bbx[2]


    def bbxScale(self, im_size, scale=1.3):
        assert(scale > 1)
        x = np.around(max(1, self.x - (scale * self.w - self.w) / 2.0))
        y = np.around(max(1, self.y - (scale * self.h - self.h) / 2.0))
        w = np.around(min(scale * self.w, im_size[1] - x))
        h = np.around(min(scale * self.h, im_size[0] - y))
        return BBox([int(x), int(x+w), int(y), int(y+h)])

    def bbxShift(self, im_size, shift=0.03):
        direction = np.random.randn(2)
        x = np.around(max(1, self.x - self.w * shift * direction[0]))
        y = np.around(max(1, self.y - self.h * shift * direction[1]))
        w = min(self.w, im_size[1] - x)
        h = min(self.h, im_size[0] - y)
        return BBox([x, x+w, y, y+h])

    def normalizeLmToBbx(self, landmarks):
        result = []
        lmks = landmarks.copy()
        for lm in lmks:
            lm[0] = (lm[0] - self.x) / self.w
            lm[1] = (lm[1] - self.y) / self.h
            result.append(lm)
        result = np.asarray(result)
        
        return result


if __name__ == '__main__':
    data = read_data_from_txt("datasets/lfpw-trainset/", image_type = ".png")
    data2 = read_data_from_txt("datasets/helen-trainset/", image_type = ".jpg")
    #data3 = read_data_from_txt("datasets/ibug-trainset/", image_type = ".jpg")
    #data4 = read_data_from_txt("datasets/afw-trainset/", image_type = ".jpg")
    data_augmentation(data + data2, output='train_new.txt', is_training=True)

    data = read_data_from_txt("datasets/lfpw-testset/", image_type = ".png")
    data2 = read_data_from_txt("datasets/helen-testset/", image_type = ".jpg")
    data_augmentation(data + data2, output='test_new.txt', is_training=False)

