#================================== Define local file path ==================================#

img_example_path = '/data2/ai-food-pathogen-data/train/mono_1/Acquired-0.jpg'

#============================================================================================# 

import os
import numpy as np
import torch
import albumentations as A
import torchvision.transforms as tra
import cv2
import xml.etree.ElementTree as ET

## Input image dimensions for resizing
img_example = cv2.imread(img_example_path)
height, width = img_example.shape[0:2]
max_px = min(height, width)

def get_transform(train):
    if train:
        transforms = A.Compose([
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Resize(max_px, max_px),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,
                                   brightness_by_max=False, always_apply=False, p=0.7),
        ],
        keypoint_params=A.KeypointParams(format='xy'))
    else:
        transforms = A.Compose([A.Resize(max_px, max_px)],
        keypoint_params=A.KeypointParams(format='xy'))
    return transforms

class MicrobeDataset(object):
    def __init__(self, root, annot_path, transforms, n_anchors):
        self.root = root
        self.transforms = transforms
        self.n_anchors = n_anchors
        self.imgs = list(sorted(os.listdir(self.root)))
        
        point_dict = {}
        point_xy_dict = {}

        ## Dataset without lysed E. coli
        if annot_path is None:
            p_arr = np.ndarray.astype(np.array([0]), dtype=np.float32)
            for img_name in list(sorted(os.listdir(self.root))):
                point_dict[img_name] = p_arr
                point_xy_dict[img_name] = []
        
        ## Dataset with lysed E. coli
        else:
            tree = ET.parse(annot_path)
            root = tree.getroot()
            GT_points = root.findall('image')

            for i in GT_points:
                img_name = i.get('name')
                points = i.findall('points')
                num_points = 0
                
                ## Read all types of labels (e.g., 'lysis', 'non_lysis', 'non-lysis')
                xy_arr = np.zeros(shape=(2,len(points)))
                for m,j in enumerate(points):
                    if j.get('label') == 'lysis':
                        num_points = num_points+1
                        xy_arr[:,m] = j.get('points').split(',')
                xy_arr_idx = []
                for k in range(xy_arr.shape[1]):
                    if sum(xy_arr[:,k]) == 0:
                        xy_arr_idx.append(k)
                        
                ## Remove labels other than 'lysis'
                xy_arr = np.delete(xy_arr, xy_arr_idx, axis=1)
                
                ## Get the number of E. coli counts (instances) in an image
                p_arr = np.ndarray.astype(np.array([num_points]), dtype=np.float32)
                point_dict[img_name] = p_arr
                
                ## Get (x, y) coordinates for E. coli instances
                a1 = xy_arr.tolist()[0]
                a2 = xy_arr.tolist()[1]
                point_xy_dict[img_name] = list(zip(a1, a2))
                
        self.point_dict = point_dict
        self.point_xy_dict = point_xy_dict

    def __getitem__(self, idx):

        ## Get image file
        img_path = os.path.join(self.root, self.imgs[idx])
        img_name = img_path.split('/')[-1]

        ## Get image number - this is required to draw bboxes on test images
        if 'Acquired' in img_name:
            img_num = int(img_name.split('.jpg')[0].split('-')[-1])
        else:
            print('Input images should be named as Acquired-(int).jpg')


        ## Get (x, y) coordinates & E. coli counts
        if img_name in self.point_dict.keys():
            target_count = self.point_dict[img_name]
            target_xy = self.point_xy_dict[img_name]
        else:
            target = np.ndarray.astype(np.array([0]), dtype=np.float32)
            target_xy = np.ndarray.astype(np.array([0,0]), dtype=np.float32)

        ## Read input image
        if 'jpg' in img_path:
            img=cv2.imread(img_path)
            
            ## Get image dimensions to generate bounding boxes
            width = img.shape[1] # x-coord
            height = img.shape[0] # y-coord
        else:
            print('Input images need to be jpg files')

        ## Pass image and label into albumentations transforms (img.shape=[H, W, C])
        if self.transforms is not None:
            augmented = self.transforms(image=img, keypoints=target_xy)
            img = augmented["image"]
            target_xy = augmented["keypoints"]

        ## Convert image to torch tensor & normalize
        img = np.ndarray.astype(img, dtype=np.float32)
        img = tra.ToTensor()(img)
        img = (img - img.min())/(img.max() - img.min())
        
        ## Generate bounding boxes from keypoint labels 
        target_xy = np.array(target_xy)
        if target_xy.size > 0: # to avoid errors in images without lysed E. coli
            target_xy = np.concatenate((target_xy, np.zeros((len(target_xy),2))), axis=1) 
        target = {}
        bbox_a = 40
        
        if np.sum(target_xy) == 0:
            target_xy = np.zeros([0,4])
        else:
            for i in range(len(target_xy)):
                target_xy[i][0] = target_xy[i][0] - bbox_a/2 # x-coord, left
                target_xy[i][1] = target_xy[i][1] - bbox_a/2 # y-coord, bottom
                target_xy[i][2] = target_xy[i][0] + bbox_a # x-coord, right
                target_xy[i][3] = target_xy[i][1] + bbox_a # y-coord, top
                if target_xy[i][0] < 0:
                    target_xy[i][0] = 0
                if target_xy[i][1] < 0:
                    target_xy[i][1] = 0
                if target_xy[i][2] > width + bbox_a/2: # x-coord upper limit
                    target_xy[i][2] = width + bbox_a/2
                if target_xy[i][3] > height + bbox_a/2: # y-coord upper limit
                    target_xy[i][3] = height + bbox_a/2

        ## Convert targets to torch tensors 
        target_xy = torch.tensor(target_xy, dtype=torch.float)

        ## Get outputs
        target['boxes'] = torch.tensor(target_xy, dtype=torch.float32)
        target['area'] = torch.ones(len(target_xy)) * (bbox_a**2)
        target['labels'] = torch.ones((len(target_xy),), dtype=torch.int64)
        target['iscrowd'] = torch.zeros((len(target_xy),), dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])

        return img, target, idx, img_num

    def __len__(self):
        return len(self.imgs)
    