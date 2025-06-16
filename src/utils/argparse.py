import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="Pipeline Argument Parser")
    parser.add_argument('--users', type=str, default='1000', help="Jumlah user (contoh: 1000 atau 1500)")
    parser.add_argument('--top_n', type=str, default='5', help="Jumlah user yang dijelaskan (misal 5 atau 'all')")

    return parser.parse_args()