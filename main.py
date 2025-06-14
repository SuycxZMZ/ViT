# main.py
import argparse
from train import train, validate, test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'validate', 'test'])
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'validate':
        validate()
    elif args.mode == 'test':
        test()

print("Execution complete.")


