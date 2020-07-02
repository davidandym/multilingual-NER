#! /usr/bin/env python
""" Construct a label dictionary and write to a Pickle file """


# pylint: disable=invalid-name


import argparse as ap
import pickle
from ner.data.dataset import Dataset
from ner.conll import get_tag_type


if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument('--input-path', required=True)
    p.add_argument('--output-path', required=True)
    p.add_argument('--max-sentence-len', type=int, default=0)
    p.add_argument('--entity-spotting', default=False, action='store_true')
    p.add_argument('--nobio-tagging', default=False, action='store_true')
    p.add_argument('--other-tag', default='O')
    args = p.parse_args()

    if args.nobio_tagging and args.entity_spotting:    
        raise ValueError(
                f"Function acceps only one of entity-spotting or nobio-tagging")

    ds = Dataset(args.input_path, args.max_sentence_len)
    label_map = {}
    other_count = 0
    for sentence in ds.sentences:
        for label in sentence.tags:
            if label == args.other_tag:
                other_count += 1

            if args.nobio_tagging:
                tag_type = get_tag_type(label)
                if tag_type not in label_map:
                    label_map[tag_type] = len(label_map)

            elif args.entity_spotting:
                if label != args.other_tag:
                    tag_type = get_tag_type(label)
                    if tag_type not in label_map:
                        label_map[tag_type] = len(label_map)
            else:  # NER
                if label not in label_map:
                    label_map[label] = len(label_map)

    if other_count < 1:
        raise ValueError(
            f"ALERT! No instances of the assumed OTHER tag; that's weird")

    print("Label map:")
    for k, v in label_map.items():
        print(f"{k} {v}")

    print(f"Writing label map to {args.output_path}...")
    with open(args.output_path, 'wb') as handle:
        pickle.dump(label_map, handle)

    print(f"Writing plain text version to {args.output_path}.txt...")
    with open(args.output_path + ".txt", 'w') as handle:
        for k, v in label_map.items():
            handle.write(f"{k} {v}\n")
