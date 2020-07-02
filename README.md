# multilingual-NER

Code for "Sources of Transfer in Multilingual NER".

Written in python 3.6 with tensorflow 1.13.

### CRF Vocab


``` 
python scripts/pre-processing/build_crf_vocab.py --train-file /exp/jmayfield/data/ner/eng/train.iob2.txt --vocab-file outputs/crf/vocab.pickle
```


``` 
python scripts/pre-processing/build_crf_vocab.py --train-file /exp/jmayfield/data/ner/eng/train.iob2.txt --test-files /exp/jmayfield/data/ner/eng/dev.iob2.txt /exp/jmayfield/data/ner/eng/test.iob2.txt --embedding-file /exp/jmayfield/data/ner/eng/fastext.vec.txt --vocab-file outputs/crf/embedding_vocab.pickle
```



### CharNER Vocab

``` 
python scripts/pre-processing/build_label_map.py --input-path /exp/jmayfield/data/ner/eng/train.iob2.txt --output-path outputs/charner/label_map.pickle
```

``` 
python scripts/pre-processing/build_label_map.py --input-path /exp/jmayfield/data/ner/eng/train.iob2.txt --output-path outputs/charner/nobio_label_map.pickle --nobio-tagging
```

### Byte-to-span Vocab

The byte-to-span label vocabulary is hardcoded. See `byte.py :: seq2seq_indexer` for more info.
