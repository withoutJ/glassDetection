import os
import os.path

import torch.utils.data as data
from PIL import Image
import random
import numpy as np

def listdirs_only(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

class VGSDDataset(data.Dataset):
    def __init__(self, root, split="train", joint_transform = None, input_transform = None, target_transform = None):
        self.root = os.path.join(root, split)
        self.joint_transform = joint_transform
        self.input_transform = input_transform
        self.target_transform = target_transform
    
        self.input_folder = 'JPEGImages'
        self.label_folder = 'SegmentationClassPNG'
        self.img_ext = '.jpg'
        self.label_ext = '.png'

        self.num_video_frame = 0
        # get all frames from video datasets
        self.videoImg_list = self.generateImgFromVideo(self.root)


    def __getitem__(self, index):
        manual_random = random.random()  # random for transformation
        # pair in video
        exemplar_path, exemplar_gt_path, videoStartIndex, videoLength = self.videoImg_list[index]  # exemplar

        other_index = index + 1
        if other_index >= videoStartIndex + videoLength - 1: # index is the last frame in video
            other_index = videoStartIndex # select the first frame in video

        other_path, other_gt_path = self.videoImg_list[other_index]  # other

        exemplar = Image.open(exemplar_path).convert('RGB')
        other = Image.open(other_path).convert('RGB')
        exemplar_gt = Image.open(exemplar_gt_path).convert('L')
        other_gt = Image.open(other_gt_path).convert('L')

        # transformation
        if self.joint_transform is not None:
            exemplar, exemplar_gt = self.joint_transform(exemplar, exemplar_gt, manual_random, None)
            other, other_gt = self.joint_transform(other, other_gt, manual_random, None)
        if self.input_transform is not None:
            exemplar = self.input_transform(exemplar)
            other = self.input_transform(other)
        if self.target_transform is not None:
            exemplar_gt = self.target_transform(exemplar_gt)
            other_gt = self.target_transform(other_gt)
        
        sample = {'exemplar': exemplar, 'exemplar_gt': exemplar_gt, 'other': other, 'other_gt': other_gt}
        sample['exemplar_path'] = exemplar_path
        sample['other_path'] = other_path 

        return sample

    def generateImgFromVideo(self, root):
        imgs = []
        video_list = listdirs_only(os.path.join(root))
        for video in video_list:
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video, self.input_folder)) if f.endswith(self.img_ext)] # no ext
            img_list = self.sortImg(img_list)
            for img in img_list:
                # videoImgGt: (img, gt, video start index, video length)
                videoImgGt = (os.path.join(root, video,  self.input_folder, img + self.img_ext),
                        os.path.join(root, video, self.label_folder, img + self.label_ext), self.num_video_frame, len(img_list))
                imgs.append(videoImgGt)
            self.num_video_frame += len(img_list)

        return imgs
    
    def sortImg(self, img_list):
        img_int_list = [int(f) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]

    def __len__(self):
        return len(self.videoImg_list)//2*2