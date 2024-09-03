import os
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms import v2
import random


class DescriptionScene(Dataset):
    def __init__(self, data_description_path, mem, data_scene_path):
        self.description_path = data_description_path
        self.data_pov_path = data_scene_path
        available_data = open('available_data/available_data_3dfront.txt', 'r')
        self.samples = [x[:-1] for x in available_data.readlines()]
        self.mem = mem
        if self.mem:
            print('Data Loading ...')

            print('Loading descriptions ...')
            if os.path.exists('available_features/3dfront/descs_3dfront.pkl'):
                pickle_file = open('available_features/3dfront/descs_3dfront.pkl', 'rb')
                self.descs = pickle.load(pickle_file)
                pickle_file.close()
            else:
                self.descs = []
                for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                    self.descs.append(torch.load(self.description_path + os.sep + s + '.pt'))
                pickle_file = open('available_data/3dfront/descs.pkl', 'wb')
                pickle.dump(self.descs, pickle_file)
                pickle_file.close()

            print('Loading POVs ...')
            if self.data_pov_path is not None:
                if os.path.exists('available_features/3dfront/pov_images_3dfront.pkl'):
                    pickle_file = open('available_features/3dfront/pov_images_3dfront.pkl', 'rb')
                    self.pov_images = pickle.load(pickle_file)
                    pickle_file.close()
                else:
                    self.pov_images = []
                    for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                        self.pov_images.append(torch.load(self.data_pov_path + os.sep + s + '.pt'))
                    pickle_file = open('available_features/3dfront/pov_images_3dfront.pkl', 'wb')
                    pickle.dump(self.pov_images, pickle_file)
                    pickle_file.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mem:
            desc_tensor = self.descs[index]
            scene_img_tensor = self.pov_images[index]
        else:
            desc_tensor = torch.load(self.description_path + os.sep + self.samples[index] + '.pt')
            scene_img_tensor = torch.load(self.data_pov_path + os.sep + self.samples[index] + '.pt')

        return desc_tensor, scene_img_tensor, index


class DescriptionSceneMuseum(Dataset):
    def __init__(self, data_description_path, mem, data_scene_path):
        self.description_path = data_description_path
        self.data_pov_path = data_scene_path
        available_data = open('available_data/available_data_museum.txt', 'r')
        self.samples = [x[:-1] for x in available_data.readlines()]
        self.mem = mem
        if self.mem:
            print('Data Loading ...')

            print('Loading descriptions ...')
            if os.path.exists('available_features/museums/descs_museum.pkl'):
                pickle_file = open('available_features/museums/descs_museum.pkl', 'rb')
                self.descs = pickle.load(pickle_file)
                pickle_file.close()
            else:
                self.descs = []
                for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                    self.descs.append(torch.load(self.description_path + os.sep + s + '.pt'))
                pickle_file = open('available_features/museums/descs_museum.pkl', 'wb')
                pickle.dump(self.descs, pickle_file)
                pickle_file.close()

            print('Loading POVs ...')
            if self.data_pov_path is not None:
                if os.path.exists('available_features/museums/pov_images_museum.pkl'):
                    pickle_file = open('available_features/museums/pov_images_museum.pkl', 'rb')
                    self.pov_images = pickle.load(pickle_file)
                    pickle_file.close()
                else:
                    self.pov_images = []
                    for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                        self.pov_images.append(torch.load(self.data_pov_path + os.sep + s + '.pt'))
                    pickle_file = open('available_features/museums/pov_images_museum.pkl', 'wb')
                    pickle.dump(self.pov_images, pickle_file)
                    pickle_file.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mem:
            desc_tensor = self.descs[index]
            scene_img_tensor = self.pov_images[index]
        else:
            desc_tensor = torch.load(self.description_path + os.sep + self.samples[index] + '.pt')
            scene_img_tensor = torch.load(self.data_pov_path + os.sep + self.samples[index] + '.pt')
        return desc_tensor, scene_img_tensor, index
