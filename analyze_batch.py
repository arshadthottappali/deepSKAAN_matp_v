import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--num_species', type=int, default=4)
    args = parser.parse_args()
    
    csv_files = glob.glob(os.path.join(args.data_dir, '*.csv'))
    for f in csv_files:
        print(f"Processing {f}...")
        os.system(f"python analyze.py --data {f} --model {args.model} --num_species {args.num_species}")

if __name__ == '__main__':
    main()
