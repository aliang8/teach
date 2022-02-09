from sacred import Experiment

from seq2seq.utils import helper_util
from seq2seq.config import exp_ingredient, train_ingredient
ex = Experiment("train", ingredients=[train_ingredient, exp_ingredient])


def create_datasets():
    # Create datasets
    data_partitions = ["train", "valid_seen", "valid_unseen"]
    datasets, loaders = [], {}

    for partition in data_partitions:
        dataset = TATCDataset(args.data.train, partition, args,
                              args.data.ann_type)
        datasets.append(dataset)

    # Get PyTorch dataloaders
    loader_args = {
        "num_workers": args.num_workers,
        "drop_last": (torch.cuda.device_count() > 1),
        "collate_fn": helper_util.identity,
    }

    for dataset in datasets:
        loader = torch.utils.data.DataLoader(dataset,
                                             args.batch_size,
                                             shuffle=True,
                                             **loader_args)
        loaders[dataset.id] = loader

    # Get vocab from dataset with longest vocabulary
    vocab = sorted(datasets, key=lambda x: len(x.vocab["word"]))[-1].vocab

    for dataset in datasets:
        dataset.vocab_translate = vocab


@ex.automain
def main(exp, train):
    args = helper_util.AttrDict(**exp, **train)
    import ipdb
    ipdb.set_trace()
    datasets, data_loaders = create_datasets(args)