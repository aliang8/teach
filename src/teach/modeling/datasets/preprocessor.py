import os
import copy

import json
import revtok
from modeling.utils import data_util
from vocab import Vocab


class Preprocessor(object):
    def __init__(self, vocab, subgoal_ann=False, is_test_split=False, frame_size=300):
        self.subgoal_ann = subgoal_ann
        self.is_test_split = is_test_split
        self.frame_size = frame_size

        if vocab is None:
            self.vocab = {
                "word": Vocab(["<<pad>>", "<<seg>>", "<<goal>>", "<<mask>>"]),
                "action_low": Vocab(["<<pad>>", "<<seg>>", "<<stop>>", "<<mask>>"]),
                "action_high": Vocab(["<<pad>>", "<<seg>>", "<<stop>>", "<<mask>>"]),
            }
        else:
            self.vocab = vocab

        self.word_seg = self.vocab["word"].word2index("<<seg>>", train=False)

    @staticmethod
    def numericalize(vocab, words, train=True):
        """
        converts words to unique integers
        """
        if not train:
            new_words = set(words) - set(vocab.counts.keys())
            if new_words:
                # replace unknown words with <<pad>>
                words = [w if w not in new_words else "<<pad>>" for w in words]
        return vocab.word2index(words, train=train)

    def process_sentences(self, sentences):
        sentences = [revtok.tokenize(data_util.remove_spaces_and_lower(sent)) for sent in sentences]
        sentences = [[w.strip().lower() for w in sent] for sent in sentences]
        return sentences

    def process_language(self, ex, traj, r_idx, is_test_split=False):
        if self.is_test_split:
            is_test_split = True

        commander_utterances = []
        driver_utterances = []
        interactions = traj["tasks"][0]["episodes"][0]["interactions"]

        for interaction in interactions:
            if "utterance" in interaction:
                if interaction["agent_id"] == 0:
                    commander_utterances.append(interaction["utterance"])
                    driver_utterances.append("")
                elif interaction["agent_id"] == 1:
                    driver_utterances.append(interaction["utterance"])
                    commander_utterances.append("")
            else:
                commander_utterances.append("")
                driver_utterances.append("")

        goal_desc = traj["tasks"][0]["desc"]
        goal_desc = revtok.tokenize(data_util.remove_spaces_and_lower(goal_desc))
        goal_desc = [w.strip().lower() for w in goal_desc]
        traj["lang_goal"] = [
            self.numericalize(self.vocab["word"], goal_desc, train=not is_test_split) 
        ]

        commander_utterances_tok = self.process_sentences(commander_utterances)
        driver_utterances_tok = self.process_sentences(driver_utterances)

        commander_utterances_tok = [utter + ["<<sent>>"] for utter in commander_utterances_tok] + [["<<stop>>"]]
        driver_utterances_tok = [utter + ["<<sent>>"] for utter in driver_utterances_tok] + [["<<stop>>"]]

        traj["commander_utterance_tok"] = commander_utterances_tok
        traj["driver_utterances_tok"] = driver_utterances_tok

        traj["commander_utterances"] = [
            self.numericalize(self.vocab["word"], x, train=not is_test_split) for x in commander_utterances_tok
        ]

        traj["driver_utterances"] = [
            self.numericalize(self.vocab["word"], x, train=not is_test_split) for x in driver_utterances_tok
        ]

    def process_actions(self, ex, traj):
        # Action at each timestep is a tuple of [Commander, Follower]
        traj["actions_low"] = list()

        TEACH_SRC = os.environ["TEACH_SRC_DIR"]
        idx_to_action_json = "meta_data_files/ai2thor_resources/action_idx_to_action_name.json"
        action_to_idx_json = "meta_data_files/ai2thor_resources/action_to_action_idx.json"

        with open(os.path.join(TEACH_SRC, idx_to_action_json)) as f:
            idx_to_action_name = json.load(f)

        with open(os.path.join(TEACH_SRC, action_to_idx_json)) as f:
            action_to_idx = json.load(f)

        all_interactions = ex['tasks'][0]['episodes'][0]['interactions']
        
        # num_interactions = len(all_interactions)

        no_op_commander = dict(
            agent_id=0,
            action=self.vocab["commander_action_low"].word2index("NoOp", train=True),
            action_name="NoOp",
            success=1,
            query="",
            commander_obs="",
            driver_obs="",
            duration=1
        )

        no_op_driver = no_op_commander.copy()
        no_op_driver['agent_id'] = 1

        # Add the ID and action names
        # processed_interactions = []
        for i, action in enumerate(all_interactions):
            action_dict = action.copy()
            idx = action["action_id"]
            action_idx = action_to_idx[str(idx)] # get the actual index
            action_name = action_dict["action_name"] = idx_to_action_name[str(action_idx)] # get the action name
            key = "driver_action_low" if action_dict["agent_id"] == 1 else "commander_action_low"
            action_dict["action"] = self.vocab[key].word2index(action_name, train=True)
            if action_dict["agent_id"] == 0:
                no_op_driver['time_start'] = action_dict['time_start']
                traj["actions_low"].append([action_dict, no_op_driver])
            else:
                no_op_commander['time_start'] = action_dict['time_start']
                traj["actions_low"].append([no_op_commander, action_dict])

        # ctr = 0
        # while ctr < len(processed_interactions) - 1:
        #     action_dict = processed_interactions[ctr].copy()
        #     next_action_dict = processed_interactions[ctr+1]
        #     if action_dict["agent_id"] == 0:
        #         if next_action_dict["agent_id"] == 1:
        #             traj["actions_low"].append([action_dict, next_action_dict]) # C, F
        #             ctr += 2
        #         else:
        #             no_op_driver = no_op_driver.copy()
        #             no_op_driver['time_start'] = action_dict['time_start']
        #             traj["actions_low"].append([action_dict, no_op_driver])
        #             ctr += 1
        #     elif action_dict["agent_id"] == 1:
        #         if next_action_dict["agent_id"] == 0:
        #             traj["actions_low"].append([next_action_dict, action_dict]) # C, F
        #             ctr += 2
        #         else:
        #             no_op_commander = no_op_commander.copy()
        #             no_op_commander['time_start'] = action_dict['time_start']
        #             traj["actions_low"].append([no_op_commander, action_dict])
        #             ctr += 1

        # if traj["actions_low"][-1][0]["action"] == "NoOp" or traj["actions_low"][-1][1]["action"] == "NoOp":
        #     action_dict = processed_interactions[-1].copy()
        #     if action_dict["agent_id"] == 0:
        #         no_op_driver['time_start'] = action_dict['time_start']
        #         traj["actions_low"].append([action_dict, no_op_driver])
        #     else:
        #         no_op_commander['time_start'] = action_dict['time_start']
        #         traj["actions_low"].append([no_op_commander, action_dict])
