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
    Wrapper around Seq2Seq Model for teach inference
    """
    def __init__(self, process_index: int, num_processes: int,
                 model_args: List[str]):
        """Constructor

        :param process_index: index of the eval process that launched the model
        :param num_processes: total number of pro∏cesses launched
        :param model_args: extra CLI arguments to teach_eval will be passed along to the model
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=1, help="Random seed")
        parser.add_argument("--device",
                            type=str,
                            default="cuda",
                            help="cpu or cuda")
        parser.add_argument(
            "--commander_model_dir",
            type=str,
            required=True,
            help="Commander model folder name under $TEACH_LOGS")
        parser.add_argument("--driver_model_dir",
                            type=str,
                            required=True,
                            help="Driver model folder name under $TEACH_LOGS")
        parser.add_argument("--checkpoint",
                            type=str,
                            default="latest.pth",
                            help="latest.pth or model_**.pth")
        parser.add_argument("--visual_checkpoint",
                            type=str,
                            required=True,
                            help="Path to FasterRCNN model checkpoint")

        args = parser.parse_args(model_args)
        args.dout_commander = args.commander_model_dir
        args.dout_driver = args.driver_model_dir

        self.args = args

        logger.info("Seq2SeqModel using args %s" % str(args))
        np.random.seed(args.seed)

        self.model_args = None
        self.extractor = None
        self.vocab = None

        # Setup commander and driver models
        self.commander_model = self.set_up_model(process_index,
                                                 agent="commander")
        self.driver_model = self.set_up_model(process_index, agent="driver")
        self.preprocessor = Preprocessor(vocab=self.driver_model.vocab)

        self.input_dict = None
        self.cur_tatc_instance = None

    def set_up_model(self, process_index, agent="driver"):
        if agent == "commander":
            model_path = os.path.join(self.args.commander_model_dir,
                                      self.args.checkpoint)
            os.makedirs(self.args.dout_commander, exist_ok=True)
        else:
            model_path = os.path.join(self.args.driver_model_dir,
                                      self.args.checkpoint)
            os.makedirs(self.args.dout_driver, exist_ok=True)

        logger.info(f"Loading {agent} model from {model_path}")

        model_args = model_util.load_model_args(model_path)
        # TODO: fix this
        # TODO: fix params.json file path
        # dataset_info = data_util.read_dataset_info_for_inference(self.args.model_dir)
        dataset_info = data_util.read_dataset_info("tatc_preproc_testing_2")

        train_data_name = model_args.data["train"][0]
        # train_vocab = data_util.load_vocab_for_inference(self.args.model_dir, train_data_name)
        train_vocab = data_util.load_vocab("tatc_preproc_testing_2")

        # Load model from checkpoint
        if model_path is not None:
            torch.cuda.empty_cache()
            gpu_count = torch.cuda.device_count()
            logger.info(f"gpu_count: {gpu_count}")
            device = f"cuda:{process_index % gpu_count}" if self.args.device == "cuda" else self.args.device
            self.args.device = device
            logger.info(f"Loading {agent} model agent using device: {device}")
            model, self.extractor = eval_util.load_agent("seq2seq",
                                                         model_path,
                                                         dataset_info,
                                                         self.args,
                                                         test_mode=True)

        # self.vocab = {"word": train_vocab["word"], "action_low": self.model.vocab_out}

        return model

    def start_new_tatc_instance(self, tatc_instance):
        self.commander_model.reset()
        self.driver_model.reset()

        # Setup input for tatc instance and process the goal instruction
        self.input_dict = {}
        tatc_instance = self.preprocessor.process_goal_instr(
            tatc_instance, is_test_split=True)
        lang_goal = torch.tensor(tatc_instance["lang_goal"],
                                 dtype=torch.long).to(self.args.device)
        # TODO: does it matter which emb_word i use?

        # Embed the language goal
        lang_goal = self.commander_model.emb_word(lang_goal)
        self.input_dict["lang_goal_instr"] = lang_goal
        return True

    def extract_progress_check_subtask_string():
        if self.pc_result["success"]:
            return ""
        
        for subgoal in self.pc_results["subgoals"]:
            if subgoal["success"] == 1:
                continue 
            
            if subgoal["steps"][0]["success"] == 0:
                return subgoal["description"]

            for step in subgoal["steps"]:
                if step["success"] == 1: 
                    continue 

                return steps["desc"]
    
        return ""

    def get_next_action_commander(self,
                                  commander_img,
                                  driver_img,
                                  tatc_instance,
                                  prev_action,
                                  commander_img_name=None,
                                  driver_img_name=None,
                                  tatc_name=None):
        """
        Returns a commander action
        :param commander_img: PIL Image containing commander's egocentric image
        :param driver_img: PIL Image containing driver's egocentric image
        :param tatc_instance: tatc instance
        :param prev_action: One of None or a dict with keys 'commander_action', 'obj_cls', 'driver_action', and 'obj_relative_coord' 
        containing returned values from a previous call of get_next_action
        :param commander_img_name: commander image file name
        :param driver_img_name: driver image file name
        :param tatc_name: tatc instance file name
        :return action: An action name from commander actions
        :return obj_cls: Object class for search object or select oid action
        """
        # Featurize images
        commander_img_feat = self.extractor.featurize([commander_img], batch=1)
        driver_img_feat = self.extractor.featurize([driver_img], batch=1)

        self.input_dict["commander_frames"] = commander_img_feat.unsqueeze(0)
        self.input_dict["driver_frames"] = driver_img_feat.unsqueeze(0)

        with torch.no_grad():
            m_out = self.commander_model.step(
                self.input_dict,
                self.vocab,  #TODO: fix this
                prev_action=prev_action, 
                agent="commander")

        # Predicts commander action and object class
        m_pred = model_util.extract_action_preds(
            m_out,
            self.commander_model.pad,
            self.commander_model.vocab["commander_action_low"],
            clean_special_tokens=False,
            agent="commander")[0]

        action, obj_cls = m_pred["action"], m_pred["obj_cls"]

        # Dumb commander model
        if action == "Text" and hasattr(self, "pc_result"):
            latest_instr = self.extract_progress_check_subtask_string()
            latest_instr = self.preprocessor.process_sentences([latest_instr], is_test_split=True)[0]

            latest_instr = torch.tensor(latest_instr, dtype=torch.long).to(self.args.device)
            latest_instr = self.commander_model.emb_word(latest_instr)
            import ipdb; ipdb.set_trace()
            self.input_dict["lang_goal_instr"] = latest_instr

        # TODO: handle target frame + mask

        # Assume previous action succeeded if no better info available
        prev_success = True
        if prev_action is not None and "success" in prev_action:
            prev_success = prev_action["success"]

        logger.debug("Predicted action: %s, obj = %s" %
                     (str(action), str(obj_cls)))
        return action, obj_cls

    def get_next_action_driver(self,
                               commander_img,
                               driver_img,
                               tatc_instance,
                               prev_action,
                               commander_img_name=None,
                               driver_img_name=None,
                               tatc_name=None):
        """
        Returns a driver action
        :param commander_img: PIL Image containing commander's egocentric image
        :param driver_img: PIL Image containing driver's egocentric image
        :param tatc_instance: tatc instance
        :param prev_action: One of None or a dict with keys 'commander_action', 'obj_cls', 'driver_action', and 'obj_relative_coord' 
        containing returned values from a previous call of get_next_action
        :param commander_img_name: commander image file name
        :param driver_img_name: driver image file name
        :param tatc_name: tatc instance file name
        :return action: An action name from all_agent_actions
        :return obj_relative_coord: A relative (x, y) coordinate (values between 0 and 1) indicating an object in the image;
        The TEACh wrapper on AI2-THOR examines the ground truth segmentation mask of the agent's egocentric image, selects
        an object in a 10x10 pixel patch around the pixel indicated by the coordinate if the desired action can be
        performed on it, and executes the action in AI2-THOR.
        """
        commander_img_feat = self.extractor.featurize([commander_img], batch=1)
        driver_img_feat = self.extractor.featurize([driver_img], batch=1)
        self.input_dict["commander_frames"] = commander_img_feat.unsqueeze(0)
        self.input_dict["driver_frames"] = driver_img_feat.unsqueeze(0)

        with torch.no_grad():
            m_out = self.driver_model.step(self.input_dict,
                                           self.vocab,
                                           prev_action=prev_action,
                                           agent="driver")

        m_pred = model_util.extract_action_preds(
            m_out,
            self.driver_model.pad,
            self.driver_model.vocab["driver_action_low"],
            clean_special_tokens=False,
            agent="driver")[0]

        action, predicted_click = m_pred["action"], m_pred['coord']

        if not action in obj_interaction_actions:
            predicted_click = None

        logger.debug("Predicted action: %s, click = %s" %
                     (str(action), str(predicted_click)))

        # Assume previous action succeeded if no better info available
        prev_success = True
        if prev_action is not None and "success" in prev_action:
            prev_success = prev_action["success"]

        # remove blocking actions
        action = self.obstruction_detection(action, prev_success, m_out,
                                            self.driver_model.vocab_out)
        return action, predicted_click

    def get_obj_click(self, obj_class_idx, img):
        rcnn_pred = self.object_predictor.predict_objects(img)
        obj_class_name = self.object_predictor.vocab_obj.index2word(
            obj_class_idx)
        candidates = list(
            filter(lambda p: p.label == obj_class_name, rcnn_pred))
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

    def obstruction_detection(self, action, prev_action_success, m_out,
                              vocab_out):
        """
        change 'MoveAhead' action to a turn in case if it has failed previously
        """
        if action != "Forward" or prev_action_success:
            return action
        dist_action = m_out["action"][0][0].detach().cpu()
        idx_rotateR = vocab_out.word2index("Turn Right")
        idx_rotateL = vocab_out.word2index("Turn Left")
        action = "Turn Left" if dist_action[idx_rotateL] > dist_action[
            idx_rotateR] else "Turn Right"
        logger.debug("Blocking action is changed to: %s" % str(action))
        return action
