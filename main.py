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
# This script serves as the entry point for the training, validation, and testing processes.
# It uses argparse to handle command-line arguments for selecting the mode of operation.
# The train, validate, and test functions are imported from the train module.
# The script checks the mode specified by the user and calls the corresponding function.
# The script is designed to be run from the command line, allowing users to specify the mode of operation.
# The print statement at the end confirms that the execution is complete.
# The script is structured to be modular, allowing for easy expansion or modification in the future.
# The argparse library is used to parse command-line arguments, making the script flexible and user-friendly.
# The script is intended to be run in a Python environment where the train module is available.
# The main function serves as the entry point for the script, ensuring that the appropriate function is called based on user input. 

# 冲突合并测试，这一句是其他分支的修改