from sacred import Ingredient
from sacred.settings import SETTINGS

exp_ingredient = Ingredient("exp")
train_ingredient = Ingredient("train")
eval_ingredient = Ingredient("eval")
dagger_ingredient = Ingredient("dagger")

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

@exp_ingredient.config
def cfg_exp():
    # HIGH-LEVEL MODEL SETTINGS
    # where to save model and/or logs
    name = "default"
    # model to use
    model = "seq2seq"
    # which device to use
    device = "cuda"
    # number of data loading workers or evaluation processes (0 for main thread)
    num_workers = 12
    # run the code on a small chunk of data
    fast_epoch = False

    # Set this to 1 if running on a Mac and to large numbers like 250 if running on EC2
    lmdb_max_readers = 1

    # DATA SETTINGS
    data = {
        # dataset name(s) for training and validation
        "train": None,
        # what to use as annotations: {'lang', 'lang_frames', 'frames'}
        "ann_type": "lang"
    }

@train_ingredient.config
def cfg_train():
    # GENERAL TRANING SETTINGS
    # random seed
    seed = 1

    # HYPER PARAMETERS
    # batch size
    batch = 8
    # number of epochs
    epochs = 20
    # optimizer type, must be in ('adam', 'adamw')
    optimizer = "adamw"
    # L2 regularization weight
    weight_decay = 0.33
    # learning rate
    lr = 1e-4
    # weight of action loss
    action_loss_wt = 1.0
    # weight of predicting coord loss 
    action_coord_loss_wt = 1.0
    # weight of subgoal completion predictor
    subgoal_aux_loss_wt = 0
    # weight of progress monitor
    progress_aux_loss_wt = 0

    # SEQ2SEQ PARAMETERS
    # language embedding size
    demb = 768 
    # hidden layer size
    dhid = 512
    # image feature vector size
    dframe = 2500

    # DROPOUT
    # dropout rate for attention
    attn_dropout=0
    # dropout rate for actor fc 
    actor_dropout=0
    # dropout rate for LSTM hidden states
    hstate_dropout=0.3
    # dropout rate for ResNet features
    vis_dropout=0.3
    # dropout rate for concatted input features
    input_dropout=0
    # dropout frate for langauge (goal + dialogue)
    lang_dropout=0
    
    # use teacher forcing for decoder
    dec_teacher_forcing=True