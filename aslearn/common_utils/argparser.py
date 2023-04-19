import argparse

# argparse cheatsheet

parser = argparse.ArgumentParser(description='argparser')

parser.add_argument('integers', type=str, help='position variable')
parser.add_argument('integers', type=str, nargs='+',help='multiple position variable, as a list')
parser.add_argument('integers', type=int, nargs='+',help='change the type')
parser.add_argument('--oparg', type=str, help='optinal arguments')
parser.add_argument('--oparg', type=str, default='.',help='optinal arguments with default value')
parser.add_argument('--oparg', type=str, required=True, default='', help='required optinal arguments')

args = parser.parse_args()