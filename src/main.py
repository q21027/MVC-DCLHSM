import argparse
import os
import numpy as np

from src.train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        # default='LG_cross', help="models used")
                        default='LG_cross', help="models used")
    parser.add_argument('--dataset', type=str, default="ACM",
                        help='Dataset chosen to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Dataset chosen to train')
    parser.add_argument('--epoches', type=int, default=50,
                        help='Number of epochs to Stage_1 train.')
    parser.add_argument('--embed_size', type=int, default=256,#ACM:256, DBLP:64, IMDB:128
                        help='Number of epochs to Stage_1 train.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--selfLoop', type=bool, default= False,
                        help='Adj whether has selfloop or not')



    args = parser.parse_args()
    train_model = train(args)
    train_model.run()

if __name__ == "__main__":


     main()