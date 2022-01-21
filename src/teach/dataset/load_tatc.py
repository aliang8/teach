import copy
import glob
import json
import os
import pickle
import time

import numpy as np
from tqdm import tqdm
from PIL import Image

from teach.dataset.dataset import Dataset
from teach.logger import create_logger

## loading all data + obs frames and returning a merged dict

def load(data_dir, image_dir):
    games=[]
    game_files = os.listdir(data_dir)
    for game_file in tqdm(game_files[:5]):  ##change -  load all
        # print(game_file)
        f = os.path.join(data_dir, game_file)
        game = Dataset.import_json(f, version="2.0")
        interactions = game.tasks[0].episodes[0].interactions
        # print(f, image_dir)
        commander_obs = {}
        driver_obs = {}
        game_image_dir = os.path.join(image_dir, game_file.split('.')[0])
        # print(game_image_dir)
        for img_file in os.listdir(game_image_dir):
            # print(img_file)
            f = os.path.join(game_image_dir, img_file)
            # print(f)
            time_start = '.'.join(img_file.split('.')[2:4])
            if img_file.split('.')[0]=="commander":
                commander_obs[time_start] = f #Image.open(f) ##open and copy
                # print(commander_obs[time_start].size)
            if img_file.split('.')[0]=="driver":
                driver_obs[time_start] = f #Image.open(f)
                # print(driver_obs[time_start].size)
            else:          ### add other frames if needed
                # print("pass")
                pass
        for idx in range(len(interactions)):
            time_start = str(interactions[idx].time_start)
            interactions[idx].commander_obs = commander_obs[time_start]
            interactions[idx].driver_obs = driver_obs[time_start]
            # print(interactions[idx].commander_obs.size)
        game.tasks[0].episodes[0].interactions = interactions
        games.append(game)
    return games
# load(data_dir, image_dir)