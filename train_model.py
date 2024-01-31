#!/usr/bin/env python

import os, sys
from train_12ECG_classifier import train_12ECG_classifier

if __name__ == '__main__':
    # Parse arguments.
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    model_name = sys.argv[3] # model

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    print(f'Running training code for {model_name}...')

    train_12ECG_classifier(input_directory, output_directory,model_name)

    print('Done.')