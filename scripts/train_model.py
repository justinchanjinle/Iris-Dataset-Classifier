import argparse
from pathlib import Path

from src.clean_data import CleanData
from src.ingest_data import IngestData
from src.training import Training
from utils.enumerations import MODELS_NAME_LIST, Models

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train model for deployment.')

    parser.add_argument('--raw_data_dir', required=True, type=Path, help='Directory to the raw data')
    parser.add_argument('--test_size', required=False, default=0.2, help='Size if test data')
    parser.add_argument('--test_train_random_state', required=False, default=0, help='Random state seed for splitting '
                                                                                     'test and train data')
    parser.add_argument('--model', required=True, choices=MODELS_NAME_LIST, help='Pre-prepared model')
    parser.add_argument('--model_save_dir', required=True, type=Path, help='Directory to save the model')

    parsed_args = parser.parse_args()

    # TODO: separate CleanData from Training. CleanData should have its own script
    ingest_data = IngestData(parsed_args.raw_data_dir)
    clean_data = CleanData(ingest_data, parsed_args.test_size, parsed_args.test_train_random_state)

    model = Models[parsed_args.model].value

    train_model = Training(clean_data, model)
    train_model.save_model(parsed_args.model_save_dir)
