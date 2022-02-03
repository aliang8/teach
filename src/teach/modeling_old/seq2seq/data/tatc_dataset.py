from seq2seq.data.base import BaseDataset

class TATCDataset(BaseDataset):
    def __init__(self):

    def load_data(self, path):
        pass 

    def __getitem__(self, idx):

    def load_features(self, task_json):

    @staticmethod
    def load_lang(task_json):
        """
        load numericalized language from task_json
        """
        return sum(task_json["num"]["lang_instr"], [])