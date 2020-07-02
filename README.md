# multilingual-NER

Code for the models used in "[Sources of Transfer in Multilingual NER](https://www.aclweb.org/anthology/2020.acl-main.720/)", published at ACL 2020.

Written in python 3.6 with `tensorflow-1.13`.

The code is separated into 2 parts, the `ner` package which needs to be installed via `setup.py` and the `scripts` folder which contains the executables to run the models and generate the vocabularies. Oh, and it also has the official `conlleval` script, which should be able to run on the prediction outputs of any of the models here.

To setup the `ner` package, I think it suffices to run something like:
```
python setup.py [install / develop]
```
although it is recommended that this is done in a virtual environment with no other dependences named `ner`, haha.

## Data and Vocabs

All datasets are assumed to be in "almost" CoNLL03 format (minus any meta-data) where column 0 contains the words, column 1 contains the tags, and they are tab-separated, i.e.
```
উত্তর    B-GPE
রাইন I-GPE
ওয়েস্ট    I-GPE
ফালিয়া I-GPE
অথাত্    O
যে   O
রাযে O
আমরা O
বাস  O
```
To see how files are read in, see `ner/data/io.py` and `ner/data/datsets.py`.

At this time I cannot release the LORELEI datasets that were used in the work, but they are in the process of being released on LDC as of July 2020.

### Multilingual data

For my multilingual experiments, I naively concatenated the training datasets of all languages together and treated the output as one large training file. In the paper I do not use word-embeddings, but in my early experiments I did. In this case, naively concatenating datasets together may not work well if you want to ensure that overlapping words between languages do not share pre-trained embeddings, but right now the code-base is not set up for that. It can run on word-embeddings, but there are no mechanisms in place to handle embeddings from multiple languages in a nice way.

### CRF Vocab

The CRF model has the most intensive vocabulary building, because it can operate over words, subwords, characters, and bytes.

To build a vocabulary with no pre-trained word embeddings, use something like:
```bash
python scripts/pre-processing/build_crf_vocab.py \
  --train-file ${DATA_DIR}/train.iob2.txt \
  --vocab-file outputs/crf/vocab.pickle
```
If you aren't using pre-trained embeddings, then the vocabulary is only built from the training data. If you *are* using pre-trained embeddings, then the proper input looks more like:
```bash 
python scripts/pre-processing/build_crf_vocab.py \
  --train-file ${DATA_DIR}/train.iob2.txt \
  --test-files ${DATA_DIR}/dev.iob2.txt ${DATA_DIR}/test.iob2.txt \
  --embedding-file ${DATA_DIR}/whatever_embeddings.vec.txt \
  --vocab-file outputs/crf_wembeddings/embedding_vocab.pickle
```
and you will need to pass in the embeddings file, and the test files as well so that the embedding vector can be relativized to the entire vocabulary.


### CharNER Vocab

The CharNER model only requires a BIO-less label map, just to ensure consistency across different runs of the model. It can be created like this:
```bash 
python scripts/pre-processing/build_label_map.py \
  --input-path /exp/jmayfield/data/ner/eng/train.iob2.txt \
  --output-path outputs/charner/nobio_label_map.pickle \
  --nobio-tagging
```

### Byte-to-span Vocab

The byte-to-span label vocabulary is hardcoded. See `ner/byte.py :: seq2seq_indexer` for more info.

## Training and Predicting

There are 3 models in this code-base:

- `byte_to_span`
- `charner`
- `standard_word_crf`

All experiments are run through `scripts/run_experiment.py` with two modes, train and predict. Additionally, each model has a set of default hyper-parameters which are defined in the same file as the model. For example, a default parameter set for charner looks like:
```python
@Registries.hparams.register
def charner_default():
    return HParams(
        shuffle_buffer_size=40000,
        batch_size=32,
        output_dim=0,
        birnn_layers=5,
        birnn_dim=128,
        dropout_keep_prob=[0.5, 0.5, 0.5, 0.5, 0.2],
        optimizer='adam',
        beta1=0.9,
        beta2=0.999,
        use_ema=False,
        learning_rate=0.001,
        emb_dim=128,
        emb_keep_prob=0.8,
        grad_clip_norm=1,
        nobio_label_map_path=""
    )
```

To train a model using the charner model and the above set of hyper-parameters, the command would look something like:
```bash
python scripts/run_experiment.py train \
    --train-file ${DATA_DIR}/train.iob2.txt \
    --dev-file ${DATA_DIR}/dev.iob2.txt \
    --model-path outputs/charner \
    --model charner \
    --hparam-defaults charner_default \
    --hparams-str "nobio_label_map_path=outputs/charner/nobio_label_map.pickle" \
    --train-epochs 1
```
and to create predictions using the same model would look like:
```bash
python scripts/run_experiment.py predict \
    --test-file ${DATA_DIR}/test.iob2.txt \
    --model-path outputs/charner \
    --output-file outputs/charner/predictions.txt \
    --model charner \
    --hparam-defaults charner_default \
    --hparams-str "nobio_label_map_path=outputs/charner/nobio_label_map.pickle" 
```

If I wanted to override some of the default hyperparameters, but keep things mostly the same I could modify the hparams-str, for example:
```bash
python scripts/run_experiment.py train \
    --train-file ${DATA_DIR}/train.iob2.txt \
    --dev-file ${DATA_DIR}/dev.iob2.txt \
    --model-path outputs/charner \
    --model charner \
    --hparam-defaults charner_default \
    --hparams-str "nobio_label_map_path=outputs/charner/nobio_label_map.pickle,birnn_layers=7,birnn_dim=1024" \
    --train-epochs 1
```
Examples for other models and defaults are in `experiments/local`.

## BibTex

If you want to cite this code or the original paper you can use:
```
@inproceedings{mueller-etal-2020-sources,
    title = "Sources of Transfer in Multilingual Named Entity Recognition",
    author = "Mueller, David  and
      Andrews, Nicholas  and
      Dredze, Mark",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.720",
    pages = "8093--8104",
    abstract = "Named-entities are inherently multilingual, and annotations in any given language may be limited. This motivates us to consider \textit{polyglot} named-entity recognition (NER), where one model is trained using annotated data drawn from more than one language. However, a straightforward implementation of this simple idea does not always work in practice: naive training of NER models using annotated data drawn from multiple languages consistently underperforms models trained on monolingual data alone, despite having access to more training data. The starting point of this paper is a simple solution to this problem, in which polyglot models are \textit{fine-tuned} on monolingual data to consistently and significantly outperform their monolingual counterparts. To explain this phenomena, we explore the sources of multilingual transfer in polyglot NER models and examine the weight structure of polyglot models compared to their monolingual counterparts. We find that polyglot models efficiently share many parameters across languages and that fine-tuning may utilize a large number of those parameters.",
}
```
