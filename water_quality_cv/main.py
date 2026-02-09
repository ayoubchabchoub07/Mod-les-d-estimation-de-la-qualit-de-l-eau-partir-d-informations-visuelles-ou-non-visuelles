import argparse
import os
from train import train
from predict import predict


def main():
    parser = argparse.ArgumentParser(description='Pipeline qualité de l\'eau')
    parser.add_argument('--step', choices=['train','predict'], required=True)
    parser.add_argument('--dataset', default=os.path.join(os.path.dirname(__file__), 'dataset'))
    parser.add_argument('--model', default='model.h5')
    args = parser.parse_args()

    if args.step == 'train':
        train(args.dataset)
    elif args.step == 'predict':
        preds = predict(args.dataset, model_path=args.model)
        if preds is not None:
            print(preds)


if __name__ == '__main__':
    main()
