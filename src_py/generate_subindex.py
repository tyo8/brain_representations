import math
import base64
import argparse
import sys
from numpy.random import default_rng

rng = default_rng()


def _subidx_to_bytes(subidx, n_dims):
    byte_arr = bytearray(math.ceil(n_dims / 8))

    def set_bit(bit_idx):
        byte_idx = bit_idx // 8
        bit_offset = bit_idx % 8
        byte_arr[byte_idx] |= 1 << bit_offset

    for chosen_idx in subidx:
        set_bit(chosen_idx)

    return byte_arr


def subidx_to_tag(subidx, n_dims):
    byte_arr = _subidx_to_bytes(subidx, n_dims)
    return base64.urlsafe_b64encode(byte_arr).decode()


def tag_to_subidx(tag):
    byte_arr = base64.urlsafe_b64decode(tag)

    subidx = []
    for byte_idx, b in enumerate(byte_arr):
        for bit_offset in range(8):
            if b >> bit_offset & 0x1:
                subidx.append(byte_idx * 8 + bit_offset)

    return subidx


def generate_subidx_tag(n_dims=1003, sample_prop=0.8):
    num_choices = math.floor(n_dims * sample_prop)
    subidx = rng.choice(n_dims, num_choices, replace=False)
    return subidx_to_tag(subidx, n_dims)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate subindex samplings and their unique tags"
    )
    parser.add_argument(
        "-c", "--count", type=int, default=250, help="how many subindexes to generate"
    )
    parser.add_argument(
        "-d", "--dims", type=int, default=1003, help="total number of dimensions"
    )
    parser.add_argument(
        "-p",
        "--proportion",
        type=float,
        default=0.8,
        help="proportion of dimensions to include in subindex",
    )
    parser.add_argument(
        "-f", "--outfile", help="file to write tags to, if absent write to stdout"
    )

    args = parser.parse_args()

    out_stream = open(args.outfile, "w") if args.outfile else sys.stdout
    with out_stream as f:
        for _ in range(args.count):
            f.write(generate_subidx_tag(n_dims=args.dims, sample_prop=args.proportion))
            f.write("\n")
