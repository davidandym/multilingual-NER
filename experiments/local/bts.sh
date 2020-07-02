python scripts/run_experiment.py train \
    --train-file ${DATA_DIR}/train.iob2.txt \
    --dev-file ${DATA_DIR}/dev.iob2.txt \
    --model-path outputs/byte-to-span \
    --model byte_to_span \
    --hparam-defaults byte_to_span_momentum \
    --train-epochs 1

python scripts/run_experiment.py predict \
    --test-file ${DATA_DIR}/test.iob2.txt \
    --model-path outputs/byte-to-span \
    --output-file outputs/byte-to-span/predictions.txt \
    --model byte_to_span \
    --hparam-defaults byte_to_span_momentum
