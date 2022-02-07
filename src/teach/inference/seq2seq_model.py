# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from modeling import constants
from modeling.datasets.tatc import TATCDataset
from modeling.datasets.preprocessor import Preprocessor
from modeling.utils import data_util, eval_util, model_util

from teach.inference.actions import obj_interaction_actions
from teach.inference.teach_model import TeachModel
from teach.logger import create_logger

logger = create_logger(__name__)


class Seq2SeqModel(TeachModel):
    """
    Wrapper around Seq2Seq Model for inference
    """

    def __init__(self, process_index: int, num_processes: int, model_args: List[str]):
        """Constructor

        :param process_index: index of the eval process that launched the model
        :param num_processes: total number of processes launched
        :param model_args: extra CLI arguments to teach_eval will be passed along to the model
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=1, help="Random seed")
        parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
        parser.add_argument("--commander_model_dir", type=str, required=True, help="Model folder name under $TEACH_LOGS")
        parser.add_argument("--driver_model_dir", type=str, required=True, help="Model folder name under $TEACH_LOGS")
        parser.add_argument("--checkpoint", type=str, default="latest.pth", help="latest.pth or model_**.pth")
        parser.add_argument("--visual_checkpoint", type=str, required=True, help="Path to FasterRCNN model checkpoint")

        args = parser.parse_args(model_args)
        args.dout_commander = args.commander_model_dir
        args.dout_driver = args.driver_model_dir

        self.args = args

        logger.info("Seq2SeqModel using args %s" % str(args))
        np.random.seed(args.seed)

        self.model_args = None
        self.commander_model = None
        self.follower_model = None
        self.extractor = None
        self.vocab = None
        self.preprocessor = None
        self.commander_model = self.set_up_model(process_index, agent='commander')
        self.driver_model = self.set_up_model(process_index, agent='driver')
        self.preprocessor = Preprocessor(vocab=self.driver_model.vocab)
        self.input_dict = None
        self.cur_tatc_instance = None

    def set_up_model(self, process_index, agent='driver'):
        if agent == 'commander':
            model_path = os.path.join(self.args.commander_model_dir, self.args.checkpoint)
            os.makedirs(self.args.dout_commander, exist_ok=True)
        else:
            model_path = os.path.join(self.args.driver_model_dir, self.args.checkpoint)
            os.makedirs(self.args.dout_driver, exist_ok=True)

        logger.info(f"Loading {agent} model from {model_path}")

        model_args = model_util.load_model_args(model_path)
        # dataset_info = data_util.read_dataset_info_for_inference(self.args.model_dir)
        dataset_info = data_util.read_dataset_info("tatc_dataset")

        train_data_name = model_args.data["train"][0]
        # train_vocab = data_util.load_vocab_for_inference(self.args.model_dir, train_data_name)
        train_vocab = data_util.load_vocab("tatc_dataset")

        # self.object_predictor = eval_util.load_object_predictor(self.args)
        if model_path is not None:
            torch.cuda.empty_cache()
            gpu_count = torch.cuda.device_count()
            logger.info(f"gpu_count: {gpu_count}")
            device = f"cuda:{process_index % gpu_count}" if self.args.device == "cuda" else self.args.device
            self.args.device = device
            logger.info(f"Loading {agent} model agent using device: {device}")
            model, self.extractor = eval_util.load_agent("seq2seq", model_path, dataset_info, self.args, test_mode=True)

        # self.vocab = {"word": train_vocab["word"], "action_low": self.model.vocab_out}

        return model

    def start_new_tatc_instance(self, tatc_instance, tatc_name=None):
        self.commander_model.reset()
        self.driver_model.reset()

        self.input_dict = {}
        tatc_instance = self.preprocessor.process_goal_instr(tatc_instance)
        lang_goal = torch.tensor(tatc_instance["lang_goal"], dtype=torch.long).to(self.args.device)
        lang_goal = self.commander_model.emb_word(lang_goal)
        self.input_dict["lang_goal_instr"] = lang_goal

        return True

    #     self.cur_tatc_instance = data_util.process_traj(
    #         tatc_instance, Path(os.path.join("test", tatc_instance["instance_id"])), 0, self.preprocessor
    #     )

    #     lang = TATCDataset.load_lang(self.cur_tatc_instance)
    #     feat_numpy = {"commander_utterances": lang[0], "driver_utterances": lang[1]}
    #     _, self.input_dict, _ = data_util.tensorize_and_pad(
    #         [(self.cur_tatc_instance, feat_numpy)], self.args.device, constants.PAD
    #     )

        # if not self.args.skip_tatc_history and tatc_history_images is not None and len(tatc_history_images) > 0:
        # img_features = self.extractor.featurize(tatc_history_images, batch=32)
        # self.model.frames_traj = img_features
        # self.model.frames_traj = torch.unsqueeze(self.model.frames_traj, dim=0)
        # self.model.action_traj = torch.tensor(
        #     [
        #         self.vocab["action_low"].word2index(action["action_name"])
        #         for action in tatc_instance["driver_action_history"]
        #     ],
        #     device=self.args.device,
        # )
        # self.model.action_traj = torch.unsqueeze(self.model.action_traj, 0)
        # return True

    def get_next_action_commander(self, img, tatc_instance, prev_action, img_name=None, tatc_name=None):
        img_feat = self.extractor.featurize([img], batch=1)

        # Input dict should have `lang_goal_instr` and `frames`
        self.input_dict["frames"] = img_feat.unsqueeze(0)

        with torch.no_grad():
            prev_api_action = None
            if prev_action is not None and "commander_action" in prev_action:
                prev_api_action = prev_action["commander_action"]

            m_out = self.commander_model.step(self.input_dict, self.vocab, prev_action=prev_api_action)
        
        # Predicts action and obj cls
        m_pred = model_util.extract_action_preds_commander(
            m_out, self.commander_model.pad, self.commander_model.vocab["commander_action_low"], clean_special_tokens=False
        )[0]

        # Assume previous action succeeded if no better info available
        prev_success = True
        if prev_action is not None and "success" in prev_action:
            prev_success = prev_action["success"]

        action, obj_cls = m_pred["action"], m_pred["obj_cls"]
        return action, obj_cls

    def get_next_action_driver(self, img, tatc_instance, prev_action, img_name=None, tatc_name=None):
        """
        Returns a driver action
        :param img: PIL Image containing agent's egocentric image
        :param tatc_instance: tatc instance
        :param prev_action: One of None or a dict with keys 'action' and 'obj_relative_coord' containing returned values
        from a previous call of get_next_action
        :param img_name: image file name
        :param tatc_name: tatc instance file name
        :return action: An action name from all_agent_actions
        :return obj_relative_coord: A relative (x, y) coordinate (values between 0 and 1) indicating an object in the image;
        The TEACh wrapper on AI2-THOR examines the ground truth segmentation mask of the agent's egocentric image, selects
        an object in a 10x10 pixel patch around the pixel indicated by the coordinate if the desired action can be
        performed on it, and executes the action in AI2-THOR.
        """
        # import ipdb; ipdb.set_trace()
        img_feat = self.extractor.featurize([img], batch=1)
        self.input_dict["frames"] = img_feat.unsqueeze(0)

        with torch.no_grad():
            prev_api_action = None
            if prev_action is not None and "driver_action" in prev_action:
                prev_api_action = prev_action["driver_action"]
            m_out = self.driver_model.step(self.input_dict, self.vocab, prev_action=prev_api_action)

        m_pred = model_util.extract_action_preds_driver(
            m_out, self.driver_model.pad, self.driver_model.vocab["driver_action_low"], clean_special_tokens=False
        )[0]

        action, predicted_click = m_pred["action"], m_pred['coord']

        # obj = None
        # if action in obj_interaction_actions and len(m_pred["object"]) > 0 and len(m_pred["object"][0]) > 0:
        if action in obj_interaction_actions:
            # obj = m_pred["object"][0][0]
            pass
        else:
            predicted_click = None

        logger.debug("Predicted action: %s, click = %s" % (str(action), str(predicted_click)))

        # Assume previous action succeeded if no better info available
        prev_success = True
        if prev_action is not None and "success" in prev_action:
            prev_success = prev_action["success"]

        # remove blocking actions
        action = self.obstruction_detection(action, prev_success, m_out, self.driver_model.vocab_out)
        return action, predicted_click

    def get_obj_click(self, obj_class_idx, img):
        rcnn_pred = self.object_predictor.predict_objects(img)
        obj_class_name = self.object_predictor.vocab_obj.index2word(obj_class_idx)
        candidates = list(filter(lambda p: p.label == obj_class_name, rcnn_pred))
        if len(candidates) == 0:
            return [np.random.uniform(), np.random.uniform()]
        index = np.argmax([p.score for p in candidates])
        mask = candidates[index].mask[0]
        predicted_click = list(np.array(mask.nonzero()).mean(axis=1))
        predicted_click = [
            predicted_click[0] / mask.shape[1],
            predicted_click[1] / mask.shape[0],
        ]
        return predicted_click

    def obstruction_detection(self, action, prev_action_success, m_out, vocab_out):
        """
        change 'MoveAhead' action to a turn in case if it has failed previously
        """
        if action != "Forward" or prev_action_success:
            return action
        dist_action = m_out["action"][0][0].detach().cpu()
        idx_rotateR = vocab_out.word2index("Turn Right")
        idx_rotateL = vocab_out.word2index("Turn Left")
        action = "Turn Left" if dist_action[idx_rotateL] > dist_action[idx_rotateR] else "Turn Right"
        logger.debug("Blocking action is changed to: %s" % str(action))
        return action
