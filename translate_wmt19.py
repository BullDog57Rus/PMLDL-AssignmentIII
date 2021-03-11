from argparse import ArgumentParser

import torch

parser = ArgumentParser(prog='translate_wmt19.py', description='Translate given text.')
parser.add_argument('model_path', help='path to the trained model.')
parser.add_argument('in_data_path', help='path to the input data.')
parser.add_argument('out_data_path', help='path where to save the results.')
args = parser.parse_args()

ru2en = torch.hub.load(
    'pytorch/fairseq',
    'transformer.wmt19.ru-en',
    checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
    tokenizer='moses',
    bpe='fastbpe',
)
ru2en.eval()

print('Model was downloaded!')

inputs = open(args.in_data_path, 'r').readlines()
with open(args.out_data_path, 'w') as output_file:
    for index, line in enumerate(inputs):
        output_file.write(ru2en.translate(line) + '\n')
        print(f'Translated sentence #{index}')
