python scripts/run_experiment.py train \
    --train-file ${DATA_DIR}/train.iob2.txt \
    --dev-file ${DATA_DIR}/dev.iob2.txt \
    --model-path outputs/charner \
    --model charner \
    --hparam-defaults charner_default \
    --hparams-str "nobio_label_map_path=outputs/charner/nobio_label_map.pickle" \
    --train-epochs 1


python scripts/run_experiment.py predict \
    --test-file ${DATA_DIR}/test.iob2.txt \
    --model-path outputs/charner \
    --output-file outputs/charner/predictions.txt \
    --model charner \
    --hparam-defaults charner_default \
    --hparams-str "nobio_label_map_path=outputs/charner/nobio_label_map.pickle" 
