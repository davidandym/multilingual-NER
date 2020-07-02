""" Dataset objects / Generators / Inferencers """


# pylint: disable=too-many-lines,invalid-name,no-self-use,too-many-locals


import tensorflow as tf

from ner.data.io import load_conll_data
from ner.hparams import HParams
import ner.byte as byte


BASE_TAG = 'O'
SPACE_BYTE = bytes(" ", encoding='utf-8')[0]


# ----------------------------------
# Generic Classes
# ----------------------------------


class Sentence():
    """ Sentence class:
        Sentence is composed of words and the tags of those words.

        Should be able to:
            - Return words and tags
            - Return itself as a sequence of bytes and the associated tags for
              each byte, as well as spans.
    """
    def __init__(self, _id, word_list, tag_list=None):
        self.id = _id
        self.word_list = word_list
        if tag_list is None:
            self.tag_list = [BASE_TAG for _ in word_list]
        else:
            self.tag_list = tag_list

        wbytes, tbytes = byte.convert_sentence_into_byte_sequence(
            self.word_list,
            self.tag_list,
            space_idx=SPACE_BYTE,
            other=BASE_TAG
        )
        self.byte_sequence = wbytes
        self.byte_sequence_tags = tbytes

    def get_byte_and_tag_sequence(self):
        """ return 1 sequence of bytes, representing the sentences, as well as
        a sequence of tags which represent the label for each byte.
        """
        return self.byte_sequence, self.byte_sequence_tags

    def byte_len(self):
        """ Return the length of the sentence represented by it's byte-sequence
        """
        return len(self.byte_sequence_tags)

    def num_words(self):
        """ Number of words """
        return len(self.word_list)

    @property
    def words(self):
        """ Words in a sentence. """
        return self.word_list

    @property
    def tags(self):
        """ Word-level tags in a sentence. """
        return self.tag_list


    def __repr__(self):
        return ("Sentence(ByteLen: {}\n\tWords: {}\n\tTags: {}"
                "\n\tBytes: {}\n\tByteTags: {})").format(self.byte_len(),
                                                         self.word_list,
                                                         self.tag_list,
                                                         self.byte_sequence,
                                                         self.byte_sequence_tags)


class Document():
    """ Document class:
        Documents are composed of a number of sentences.

        Class provides:
            - Creates Sentence Objects for each sentence in the dataset, along
              with IDs, and keeps them in one place.
            - Statistics
    """

    def __init__(self, doc_id, sentence_list, doc_label):

        self.doc_id = doc_id
        sentences = sentence_list
        self.sentences = []
        self.num_words = 0
        self.num_bytes = 0
        self.num_sentences = 0
        self.doc_label_list = None

        if sentences:
            for sentence in sentences:
                word_list = [wt[0] for wt in sentence]
                tag_list = [wt[1] for wt in sentence]

                sent_o = Sentence(self.num_sentences, word_list, tag_list)
                self.num_words += sent_o.num_words()
                self.num_bytes += sent_o.byte_len()
                self.num_sentences += 1
                self.sentences.append(sent_o)
                word_list = []
                tag_list = []

        if doc_label:
            self.doc_label_list = [doc_label]
        self.log_document_stats()

    def get_sentences(self):
        """ List all sentences in the document """
        return self.sentences

    def get_label(self):
        """ Get all labels of the document """
        return self.doc_label_list

    def log_document_stats(self):
        """ Just log some basic info """
        tf.logging.info("Document Statistics for {}\n\
                        Total number of words: {}\n\
                        Total number of bytes: {}\n\
                        Total number of sentences: {}".format(
                            self.doc_id,
                            self.num_words,
                            self.num_bytes,
                            self.num_sentences))


class Dataset():
    """ Dataset class:
        Datasets are composed of a number of sentences.

        *Making the assumption that sentences are the "end goal" of our NER
        systems. What we care about is writing out predictions in
        sentence-level chunks, and we don't need to break things down further
        at this level.

        Class provides:
            - Creates Sentence Objects for each sentence in the dataset, along
              with IDs, and keeps them in one place.
            - Statistics
    """

    def __init__(self, file_name, max_sentence_len=0):

        self.file_name = file_name
        sentences = load_conll_data(file_name, max_sentence_len)
        self.sentences = []
        self.documents = []
        self.num_words = 0
        self.num_bytes = 0
        self.num_documents = 0
        self.num_sentences = 0

        if '-DOCSTART-' in sentences[0]:
            #if the first line has a DOCSTART, then treat as doc-separated CoNLL file
            if len(sentences[0].split('\t')) > 1:
                #labels are tab separated, same line as DOCSTART
                label = sentences[0].split('\t')[1]
            else:
                label = None
            sentence_list = [] #for each document
            for line in sentences:
                if '-DOCSTART-' in line:
                    #put everything into a new Document object
                    if sentence_list: #skip creating first empty Document object
                        curr_doc = Document(self.num_documents, sentence_list, label)
                        self.num_documents += 1
                        self.documents.append(curr_doc)
                        self.create_sentences(sentence_list)
                        sentence_list = []
                        if len(line.split('\t')) > 1:
                            label = line.split('\t')[1]
                else:
                    sentence_list.append(line)

            if sentence_list: #create and add in last document
                curr_doc = Document(self.num_documents, sentence_list, label)
                self.num_documents += 1
                self.documents.append(curr_doc)
                self.create_sentences(sentence_list)

        else: #else, treat as normal
            self.create_sentences(sentences)

        self.log_dataset_stats()

    def create_sentences(self, sentences):
        """ Build a sentence object and add it to the object's dataset. """
        for sentence in sentences:
            word_list = [wt[0] for wt in sentence]
            tag_list = [wt[1] for wt in sentence]

            sent_o = Sentence(self.num_sentences, word_list, tag_list)
            self.num_words += sent_o.num_words()
            self.num_bytes += sent_o.byte_len()
            self.num_sentences += 1
            self.sentences.append(sent_o)

    def get_dataset_as_byte_sequence_and_spans(self):
        """ Returns the entire dataset as a sequence of bytes, and the
        tag-spans for that sequence. Spans are returned in a map where the key
        is the absolute start position of that span.

        (for byte2span)
        """
        byte_sequence = []
        tag_sequence = []

        for sentence in self.get_sentences():
            bites, tags = sentence.get_byte_and_tag_sequence()

            byte_sequence += bites
            tag_sequence += tags

            byte_sequence += [SPACE_BYTE]
            tag_sequence += [BASE_TAG]

        assert len(byte_sequence) == len(tag_sequence)

        spans = byte.build_spans_from_tags(tag_sequence, other=BASE_TAG)

        span_start_map = {}

        for span in spans:
            assert span.start not in span_start_map
            span_start_map[span.start] = span

        tf.logging.info(("Done prepping B2S data for {}...\n"
                         "\t{} Bytes in sequence\n"
                         "\t{} Spans in sequence").format(self.file_name,
                                                          len(byte_sequence),
                                                          len(span_start_map)))
        return byte_sequence, span_start_map

    def get_sentences(self):
        """ List all sentences in the dataset """
        return self.sentences

    def get_documents(self):
        """ Return a list of all documents in the datase """
        return self.documents

    def log_dataset_stats(self):
        """ Just log some basic info """
        tf.logging.info("Dataset Statistics for {}\n\
                        Total number of words: {}\n\
                        Total number of bytes: {}\n\
                        Total number of sentences: {}".format(
                            self.file_name,
                            self.num_words,
                            self.num_bytes,
                            self.num_sentences))


class Generator():
    """ Generator class should take a dataset, and whatever other parameters it
    might need.

    Will need to implement:
        - iterate: an iterative function that yields data examples that
                    conform to the shapes and types returned by datashape and
                    datatypes
        - datashape: Return a tf.Dataset Generator acceptable shape
        - datatypes: Return a tf.Dataset Generator acceptable shape
                      corresponding to the types of the data
        - estimator_params: Return a dict of params for a tf.estimator
    """

    def __init__(self, dataset: Dataset, hp: HParams):
        self.dataset = dataset
        self.hp = hp

    def generator(self):
        """ Returns a dataset generator, to be passed into a tf.Dataset
        from_generator method.
        """
        return self.iterate

    def iterate(self):
        """ Implement in a subclass. This function should be an iterative
        function, and should yield dataset elements that conform to it's
        datashape and datatypes methods.
        """
        raise Exception("Needs to be sublassed")

    def datashape(self):
        """ Return the shape of the dataset elements, for a tf.Dataset """
        raise Exception("Needs to be sublassed")

    def datatypes(self):
        """ Return the type of the dataset elements, for a tf.Dataset """
        raise Exception("Needs to be sublassed")

    def estimator_params(self):
        """ Return tf.estimator parameters to be used with this generator."""
        raise Exception("Needs to be sublassed")



class Predictor():
    """ Predictor class.

    predictor.output_predictions(outputfile) will output sentence-level CoNLL
    style predictions to outputfile, suitable for evaluation with the standard
    conlleval script.

    Subclasses need to implement:
        - gather: Convert predictions into sentence-level tuples of the form
                    [(word, gold_tag, predicted_tag)] and save to sentence
                    predictions.
    """
    def __init__(self, dataset, hp):
        self.dataset = dataset
        self.hp = hp
        self.sentence_predictions = []

    def gather(self, predictions):
        """ Subclasses should take some form of predictions, and store the
        sentence-level word tags in self.sentence_predictions.
        """
        raise Exception("Need to be subclassed!")

    def output_predictions(self, output_file):
        """ Print out word-level tags in CoNLL format to output_file. """
        outfile = open(output_file, 'w', encoding='utf-8')
        for sentence_pred in self.sentence_predictions:
            for word, gold, pred in sentence_pred:
                outfile.write("{} {} {}\n".format(word, gold, pred))
            outfile.write("\n")
        outfile.close()
