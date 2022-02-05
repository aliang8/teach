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
from teach.dataset.task_THOR import Task_THOR

## loading all data + obs frames and returning a merged dict


def add_images_path(base_dir, phase):
    print(f"Processing {base_dir}/{phase}")
    games = []

    data_dir = os.path.join(base_dir, "games_with_all_file_paths", phase)
    images_dir = os.path.join(base_dir, "prog_check_files", phase, "PC")
    # images_dir = os.path.join(base_dir, "images", phase)
    new_data_dir = os.path.join(base_dir, "games_with_all_file_paths", phase)

    game_files = os.listdir(data_dir)

    for game_file in tqdm(game_files):
        f = os.path.join(data_dir, game_file)
        with open(f) as h:
            game = json.load(h)
        interactions = game["tasks"][0]["episodes"][0]["interactions"]

        commander_obs = {}
        driver_obs = {}
        targetobject = {}
        pc_status = {}
        game_image_dir = os.path.join(images_dir, game_file.split(".")[0])

        for img_file in os.listdir(game_image_dir):
            img_f = os.path.join(game_image_dir, img_file)
            time_start = ".".join(img_file.split(".")[2:4])
            if img_file.split(".")[0] == "commander":
                commander_obs[time_start] = img_f 
            elif img_file.split(".")[0] == "driver":
                driver_obs[time_start] = img_f 
            elif img_file.split(".")[0] == "targetobject":
                targetobject[time_start] = img_f
            elif img_file.split(".")[1] == "status":
                pc_status[time_start] = img_f
            else:  ### add other frames if needed
                pass
        # import ipdb; ipdb.set_trace()
        for idx in range(len(interactions)):
            time_start = str(interactions[idx]["time_start"])
            # interactions[idx].commander_obs = commander_obs[time_start]
            # interactions[idx].driver_obs = driver_obs[time_start]
            try:
                if interactions[idx]["action_id"]==500:
                    interactions[idx]["pc_json"] = pc_status[time_start]
            except:
                print(f)
            # if interactions[idx]["action_id"]>500:
            #     interactions[idx]["targetobject"] = targetobject[time_start]
        game["tasks"][0]["episodes"][0]["interactions"] = interactions
        new_f = os.path.join(new_data_dir, os.path.basename(f))
        with open(new_f, 'w') as h:
            json.dump(game, h)
        
        # game.dump(new_f)


if __name__ == "__main__":
    base_data_dir = "/data/anthony/teach"
# "train", 
    for phase in ["valid_seen", "valid_unseen"]:
        add_images_path(base_data_dir, phase)
