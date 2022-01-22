teach_inference \
    --data_dir /data/ishika/teach_new/teach/data \
    --output_dir /data/ishika/teach/output \
    --split train \
    --num_processes 1 \
    --metrics_file /data/ishika/teach_new/teach/output/metric \
    --model_module teach.inference.sample_model \
    --model_class SampleModel \
    --images_dir /data/ishika/teach_new/teach/output/image