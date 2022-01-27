import copy

import json
import revtok
from alfred.utils import data_util
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

    def process_language(self, ex, traj, r_idx, is_test_split=False):
        if self.is_test_split:
            is_test_split = True

        instr_anns = []

        for interaction in traj["tasks"][0]["episodes"][0]["interactions"]:
            if "utterance" in interaction:
                instr_anns.append(interaction["utterance"])

        goal_desc = traj["tasks"][0]["desc"]
        goal_desc = revtok.tokenize(data_util.remove_spaces_and_lower(goal_desc))
        goal_desc = [w.strip().lower() for w in goal_desc]

        # instr_anns = [utterance for (speaker, utterance) in ex["dialog_history"]]
        instr_anns = [revtok.tokenize(data_util.remove_spaces_and_lower(instr_ann)) for instr_ann in instr_anns]
        instr_anns = [[w.strip().lower() for w in instr_ann] for instr_ann in instr_anns]
        traj["ann"] = {
            "instr": [instr_ann + ["<<instr>>"] for instr_ann in instr_anns],
        }
        traj["ann"]["instr"] += [["<<stop>>"]]
        if "num" not in traj:
            traj["num"] = {}
        traj["num"]["lang_instr"] = [
            self.numericalize(self.vocab["word"], x, train=not is_test_split) for x in traj["ann"]["instr"]
        ]
        traj["num"]["lang_goal"] = [
            self.numericalize(self.vocab["word"], goal_desc, train=not is_test_split) 
        ]

    def tokenize_and_numericalize(self, dialog_history, numericalize=True, train=False):
        instr_anns = [utterance for (speaker, utterance) in dialog_history]

        # tokenize annotations
        instr_anns = [revtok.tokenize(data_util.remove_spaces_and_lower(instr_ann)) for instr_ann in instr_anns]

        instr_anns = [[w.strip().lower() for w in instr_ann] for instr_ann in instr_anns]
        instr = [instr_ann + ["<<instr>>"] for instr_ann in instr_anns]

        instr += [["<<stop>>"]]

        if numericalize:
            instr = [self.numericalize(self.vocab["word"], word, train=train) for word in instr]
        instr = sum(instr, [])  # flatten
        return instr

    def process_actions(self, ex, traj):
        if "num" not in traj:
            traj["num"] = {"interactions": traj['tasks'][0]['episodes'][0]['interactions']}

        traj["num"]["driver_actions_low"] = list()
        traj["num"]["commander_actions_low"] = list()

        idx_to_name_f = "/home/anthony/teach/src/teach/meta_data_files/ai2thor_resources/action_idx_to_action_name.json"
        with open(idx_to_name_f) as f:
            idx_to_name = json.load(f)

        a_to_a_f = "/home/anthony/teach/src/teach/meta_data_files/ai2thor_resources/action_to_action_idx.json"
        with open(a_to_a_f) as f:
            action_to_action_idx = json.load(f)
        
        for action in traj['tasks'][0]['episodes'][0]['interactions']:
            action_dict_with_idx = copy.deepcopy(action)

            action_idx = action_to_action_idx[str(action_dict_with_idx['action_id'])]
            action_name = idx_to_name[str(action_idx)]

            action_dict_with_idx["action"] = (self.vocab["action_low"].word2index(action_name, train=True),)
            action_dict_with_idx["action_name"] = action_name

            if action_dict_with_idx['agent_id'] == 0:
                traj["num"]["commander_actions_low"].append(action_dict_with_idx)
            else:
                traj["num"]["driver_actions_low"].append(action_dict_with_idx)
