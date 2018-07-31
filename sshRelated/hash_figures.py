#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import sys
import argparse
from hashlib import sha1
import json


def discover_figures(paths, extensions):
    def _discover_folder(folder, extensions):
        discovered_figures = list()
        for root, _, files in os.walk(folder):
            for fl in files:
                if fl.rsplit('.')[-1].lower() in extensions:
                    figpath = os.path.join(root, fl)
                    discovered_figures.append(os.path.abspath(figpath))
        return discovered_figures

    discovered_figures = list()
    for path in paths:
        disc = _discover_folder(path, extensions)
        discovered_figures += disc
    return discovered_figures


def get_figure_hash(figpath, hash_algorithm):
    if hash_algorithm.strip().lower() != 'sha1':
        raise NotImplementedError("Only sha1 hashes are supported at this time")
    hash_algorithm = lambda x: sha1(x).hexdigest()
    with open(figpath, 'br') as fid:
        data = fid.read()
        hexdigest = hash_algorithm(data)
    return hexdigest


def main():
    args.extensions = [ext.replace('.', '').lower() for ext in args.extensions]
    figures = discover_figures(args.paths, args.extensions)
    if figures:
        digest_dict = dict()
        for fig in figures:
            hexdigest = get_figure_hash(fig, 'sha1')
            digest_dict[fig] = hexdigest
        digest_json = json.dumps(digest_dict, indent=4)
        print(digest_json)
    else:
        print("No figures found", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('hash_figures', description="Yield hashes of figures.")
    parser.add_argument('-p', '--paths', help='Folder paths ot search for figures.', required=True, nargs='+')
    parser.add_argument('-e', '--extensions', help='Extensions to be regarded as figures.', required=False, nargs='+',
                        default=('.png', '.jpg', '.gif', '.pdf'))
    args = parser.parse_args()
    main()
