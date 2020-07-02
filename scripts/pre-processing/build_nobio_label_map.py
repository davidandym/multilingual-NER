#! /usr/bin/env python
""" Construct a label dictionary and write to a Pickle file """


# pylint: disable=invalid-name


import argparse as ap
import pickle

from ner.data.dataset import Dataset


if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument('--input-path', required=True)
    p.add_argument('--output-path', required=True)
    p.add_argument('--max-sentence-len', type=int, default=0)
    args = p.parse_args()
    ds = Dataset(args.input_path, args.max_sentence_len)
    label_map = {}
    for sentence in ds.sentences:
        for label in sentence.tags:
            if label.startswith("B-") or label.startswith("I-"):
                label = label[2:]
            if label not in label_map:
                label_map[label] = len(label_map)

    print("Label map:")
    for k, v in label_map.items():
        print(f"{k} {v}")

    print(f"Writing label map to {args.output_path}...")
    with open(args.output_path, 'wb') as handle:
        pickle.dump(label_map, handle)
