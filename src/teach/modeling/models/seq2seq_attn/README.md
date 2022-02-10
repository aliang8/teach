# Seq2seq Attention TATC Baseline Model 

This subdirectory is based on the seq2seq model from the ALFRED repository. The seq2seq attention model is adapted here for the TEACh TATC benchmark. Note that we have removed files not used when running seq2seq on TEACh, and many files have been significantly modified.

Below are instructions for training and evaluating commander and driver seq2seq models. If running on a laptop, it might be desirable to mimic the folder structure of the TEACh dataset, but using only a small number of games from each split, and their corresponding images and EDH instances. 


Set some useful environment variables. Optionally, you can copy these export statements over to a bash script and source it before training. 
```buildoutcfg
export TEACH_DATA=/tmp/teach-dataset
export TEACH_ROOT_DIR=/path/to/teach/repo
export TEACH_LOGS=/path/to/store/checkpoints
export VENV_DIR=/path/to/folder/to/store/venv
export TEACH_SRC_DIR=$TEACH_ROOT_DIR/src/teach
export INFERENCE_OUTPUT_PATH=/path/to/store/inference/execution/files
export MODEL_ROOT=$TEACH_SRC_DIR/modeling
export ET_ROOT=$TEACH_SRC_DIR/modeling/models/ET
export SEQ2SEQ_ROOT=$TEACH_SRC_DIR/modeling/models/seq2seq_attn
export PYTHONPATH="$TEACH_SRC_DIR:$MODEL_ROOT:$ET_ROOT:$SEQ2SEQ_ROOT:$PYTHONPATH"
```
Create a virtual environment

```buildoutcfg
python3 -m venv $VENV_DIR/teach_env
source $VENV_DIR/teach_env/bin/activate
cd TEACH_ROOT_DIR
pip install --upgrade pip 
pip install -r requirements.txt
```

Download the ET pretrained checkpoint for Faster RCNN and Mask RCNN models
```buildoutcfg
wget http://pascal.inrialpes.fr/data2/apashevi/et_checkpoints.zip
unzip et_checkpoints.zip
mv pretrained $ET_LOGS/
rm et_checkpoints.zip
```

Perform data preprocessing (this extracts image features and does some processing of game jsons). 
This step is optional as we already provide the preprocessed version of the dataset. However, we provide the command here for those who want to perform additional preprocessing. 
```buildoutcfg
python -m modeling.datasets.create_dataset \
    with args.visual_checkpoint=$TEACH_LOGS/pretrained/fasterrcnn_model.pth \
    args.data_input=games \
    args.task_type=game \
    args.data_output=tatc_dataset \
    args.vocab_path=None
```

Note: If running on laptop on a small subset of the data, use `args.vocab_path=$MODEL_ROOT/vocab/human.vocab` and add `args.device=cpu`.


Train commander and driver models (adjust the `train.epochs` value in this command to specify the number of desired train epochs).
Also see `modeling/exp_configs.py` and `modeling/models/seq2seq_attn/configs.py` for additional training parameters. You can also run `python -m modeling.train -h` to list out the parameters and their usage. 

```buildoutcfg
python -m modeling.train \
    with exp.model=seq2seq_attn \
    exp.name=seq2seq_attn_commander \
    exp.data.train=tatc_dataset \
    train.epochs=20  \
    train.seed=0
```

```buildoutcfg
python -m modeling.train \
    with exp.model=seq2seq_attn \
    exp.name=seq2seq_attn_driver \
    exp.data.train=tatc_dataset \
    train.epochs=20  \
    train.seed=0
```
Note: If running on laptop on a small subset of the data, add `exp.device=cpu` and `exp.num_workers=1`

Copy certain necessary files to the model folder so that we do not have to access training info at inference time.
```buildoutcfg
cp $TEACH_DATA/lmdb_edh/data.vocab $TEACH_LOGS/seq2seq_attn_commander
cp $TEACH_DATA/lmdb_edh/params.json $TEACH_LOGS/seq2seq_attn_commander
cp $TEACH_DATA/lmdb_edh/data.vocab $TEACH_LOGS/seq2seq_attn_driver
cp $TEACH_DATA/lmdb_edh/params.json $TEACH_LOGS/seq2seq_attn_driver
```

Evaluate the trained model
```buildoutcfg
cd $TEACH_ROOT_DIR
python src/teach/cli/inference.py \
    --model_module teach.inference.seq2seq_model \
    --model_class Seq2SeqModel \
    --data_dir $TEACH_DATA \
    --output_dir $INFERENCE_OUTPUT_PATH/inference__teach_tatc.json \
    --split valid_seen \
    --metrics_file $INFERENCE_OUTPUT_PATH/metrics__teach_tatc.json \
    --seed 0 \
    --commander_model_dir $TEACH_LOGS/seq2seq_attn_commander \
    --driver_model_dir $TEACH_LOGS/seq2seq_attn_driver \
    --object_predictor $TEACH_LOGS/pretrained/maskrcnn_model.pth \
    --device cpu
```