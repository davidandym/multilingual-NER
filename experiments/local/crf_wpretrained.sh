python scripts/run_experiment.py train \
    --train-file ${DATA_DIR}/train.iob2.txt \
    --dev-file ${DATA_DIR}/dev.iob2.txt \
    --model-path outputs/crf_wembeddings \
    --model standard_word_crf \
    --hparam-defaults word_and_byte_crf \
    --hparams-str "vocab_file=outputs/crf_wembeddings/vocab.pickle" \
    --train-epochs 1


python scripts/run_experiment.py predict \
    --test-file ${DATA_DIR}/test.iob2.txt \
    --model-path outputs/crf_wembeddings \
    --output-file outputs/crf_wembeddings/predictions.txt \
    --model standard_word_crf \
    --hparam-defaults word_and_byte_crf \
    --hparams-str "vocab_file=outputs/crf_wembeddings/vocab.pickle"
