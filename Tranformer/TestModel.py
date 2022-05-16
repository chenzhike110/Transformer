import argparse
import torch
from Modules.Transformer import Transformer

def parse_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument("--d_model", help="lattent dim", type=int, default=64)
    parse.add_argument("--q", help="Query size", type=int, default=8)
    parse.add_argument("--v", help="Value size", type=int, default=8)
    parse.add_argument("--h", help="Number of heads", type=int, default=6)
    parse.add_argument("--N", help="Number of encoder and decoder", type=int, default=4)
    parse.add_argument("--input", help="dataset input size", type=int, default=38)
    parse.add_argument("--output", help="output size", type=int, default=8)
    parse.add_argument("--dropout", help="dropout rate", type=float, default=0.3)
    parse.add_argument("--pe", help="Position embedding method", type=str, default="regular")
    args = parse.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transformer = Transformer(args.input, args.d_model, args.output, args.q, args.v, args.h, args.N, args.dropout, args.pe).to(device)
    print(transformer)