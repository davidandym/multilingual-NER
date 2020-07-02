python scripts/run_experiment.py train \
    --train-file ${DATA_DIR}/train.iob2.txt \
    --dev-file ${DATA_DIR}/dev.iob2.txt \
    --model-path outputs/crf \
    --model standard_word_crf \
    --hparam-defaults byte_level_crf \
    --hparams-str "vocab_file=/exp/dmueller/crf-ner/bytes/mono/eng/vocab.pickle" \
    --train-epochs 1


python scripts/run_experiment.py predict \
    --test-file ${DATA_DIR}/test.iob2.txt \
    --model-path outputs/crf \
    --output-file outputs/crf/predictions.txt \
    --model standard_word_crf \
    --hparam-defaults byte_level_crf \
    --hparams-str "vocab_file=/exp/dmueller/crf-ner/bytes/mono/eng/vocab.pickle"
