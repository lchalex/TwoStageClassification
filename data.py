import os
import sys
import random
import os.path as osp
import numpy as np
import pandas as pd
from collections import Counter

from PIL import Image
from glob import glob
import torch.utils.data as data
from torchvision.transforms import transforms

class BasicDataset(data.Dataset):
    def __init__(self, root, partition, transform=None):
        self.root = root
        self.partition = partition
        self.transform = transform
        class_df = pd.read_csv(osp.join(root, 'class.csv'))
        id2class = dict(zip(class_df['id'], class_df['class_id']))
        self.image_path = []
        self.label = []
        if partition == 'train':
            for key in id2class.keys():
                files = glob(osp.join(root, 'train', key, '*.JPEG'))
                self.image_path.extend(files)
                self.label.extend([id2class[key]] * len(files))
        
        elif partition == 'valid':
            val_df = pd.read_csv(osp.join(root, 'val_gt.csv'))
            for file, c in zip(val_df['file'], val_df['class_id']):
                self.image_path.append(osp.join(root, 'valid', file))
                self.label.append(c)
        
        else:
            raise Exception('Unepected partition type : {}'.format(partition))
            
    def __getitem__(self, index):
        img = self.pull_tensor(index)
        lbl = self.label[index]
        return img, lbl
    
    def __len__(self):
        assert len(self.image_path) == len(self.label)
        return len(self.label)
    
    def pull_tensor(self, index):
        img = Image.open(self.image_path[index]).convert('RGB')
        if self.transform is not None:
            torch_img = self.transform(img)
        else:
            torch_img = transforms.Compose([transforms.ToTensor()])(img)
        return torch_img

class TwoStageDataset(data.Dataset):
    def __init__(self, root, partition, transform=None):
        self.root = root
        self.partition = partition
        self.transform = transform
        class_df = pd.read_csv(osp.join(root, 'class.csv'))
        idmapper = dict()
        for key, v1, v2, v3 in zip(class_df['id'], class_df['cluster_id'], class_df['intra_cluster_id'], class_df['class_id']):
            idmapper[key] = (v1, v2, v3) # (clu_id, intra_clu_id, cls_id)

        self.image_path = []
        self.label = []
        self.cluster = []
        if partition == 'train':
            for key in idmapper.keys():
                files = glob(osp.join(root, 'train', key, '*.JPEG'))
                self.image_path.extend(files)
                self.label.extend([idmapper[key]] * len(files))
                self.cluster.extend([idmapper[key][0]] * len(files))
        
        elif partition == 'valid':
            val_df = pd.read_csv(osp.join(root, 'val_gt.csv'))
            for file, i in zip(val_df['file'], val_df['id']):
                self.image_path.append(osp.join(root, 'valid', file))
                self.label.append(idmapper[i])
                self.cluster.append(idmapper[i][0])
        
        else:
            raise Exception('Unepected partition type : {}'.format(partition))
            
    def __getitem__(self, index):
        img = self.pull_tensor(index)
        lbl = self.label[index]
        return img, lbl
    
    def __len__(self):
        assert len(self.image_path) == len(self.label)
        return len(self.label)
    
    def pull_tensor(self, index):
        img = Image.open(self.image_path[index]).convert('RGB')
        if self.transform is not None:
            torch_img = self.transform(img)
        else:
            torch_img = transforms.Compose([transforms.ToTensor()])(img)
        return torch_img

class InferenceDataset(data.Dataset):
    def __init__(self, root, partition, transform=None):
        self.root = root
        self.partition = partition
        self.transform = transform
        self.image_path = []

        if partition == 'valid':
            val_df = pd.read_csv(osp.join(root, 'val_gt.csv'))
            for file, i in zip(val_df['file'], val_df['id']):
                self.image_path.append(osp.join(root, 'valid', file))
        
        else:
            raise Exception('Unepected partition type : {}'.format(partition))
            
    def __getitem__(self, index):
        img = self.pull_tensor(index)
        return img
    
    def __len__(self):
        return len(self.image_path)
    
    def pull_tensor(self, index):
        img = Image.open(self.image_path[index]).convert('RGB')
        if self.transform is not None:
            torch_img = self.transform(img)
        else:
            torch_img = transforms.Compose([transforms.ToTensor()])(img)
        return torch_img

class ClusterBatchSampler(data.Sampler):
    def __init__(self, cluster, num_cluster_per_batch, batch_size):
        assert min(cluster) >= 0, "Cluster is must not non-negative"

        self.cluster = np.array(cluster)
        if num_cluster_per_batch > len(np.unique(cluster)):
            num_cluster_per_batch = len(np.unique(cluster))

        self.num_cluster_per_batch = num_cluster_per_batch
        self.batch_size = batch_size

        batch_indices = []
        while max(self.cluster) != -1:
            valid_cluster = self.cluster[self.cluster != -1]
            cluster_ids = np.unique(valid_cluster)
            if len(cluster_ids) <= self.num_cluster_per_batch:
                selected_cluster = cluster_ids
            else:
                prob = [len(valid_cluster[valid_cluster == i]) / len(valid_cluster) for i in cluster_ids]
                selected_cluster = np.random.choice(cluster_ids, self.num_cluster_per_batch, p=prob, replace=False)
            
            pool = np.where(np.isin(self.cluster, selected_cluster))[0]
            if len(pool) < self.batch_size:
                select = pool
            else:
                select = np.random.choice(pool, self.batch_size, replace=False)

            np.random.shuffle(select)
            batch_indices.append(select)
            self.cluster[select] = -1

            # Drop class with few instance left
            cnter = Counter(self.cluster)
            for clu, num in cnter.items():
                if num < (self.batch_size / self.num_cluster_per_batch) / 2:
                    self.cluster[self.cluster == clu] = -1
        
        self.batch_indices = batch_indices[::-1]
        self.itr = iter(self.batch_indices)

    def __iter__(self):
        return self.itr
        
    def __len__(self):
        return len(self.batch_indices)