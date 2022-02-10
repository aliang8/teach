teach_inference \
    --data_dir /data/ishika/teach_new/teach/data \
    --output_dir /data/ishika/teach/output \
    --split train \
    --num_processes 1 \
    --metrics_file /data/ishika/teach_new/teach/output/metric \
    --model_module teach.inference.sample_model \
    --model_class SampleModel \
    --images_dir /data/ishika/teach_new/teach/output/image


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python -m train \
    with exp.model=seq2seq_attn \
    exp.name=seq2seq_attn_commander \
    exp.data.train=tatc_final \
    seq2seq.epochs=20 \
    seq2seq.seed=2 \
    seq2seq.resume=False

python -m datasets.create_dataset \
    with args.visual_checkpoint=$TEACH_DATA/experiments/checkpoints/pretrained/fasterrcnn_model.pth \
    args.data_input=games_final \
    args.data_output=tatc_dataset_test \
    args.vocab_path=None \
    args.overwrite=1 \
    args.num_workers=0 \
    args.fast_epoch=True