import pandas as pd
import argparse
import matplotlib.pyplot as plt


def read_clean(args):

    df = pd.read_pickle("test_results/gaussian_noise.pkl")
    df = df.drop(["k", "m", "seed", "accuracy_gauss",
                  "mean_snr","epsilon","sample"], axis=1)
    pre_agg = df.groupby(["beta", "gamma"])
    everything = pre_agg.agg(["mean"])
    if args.matplotlib:
        df.boxplot(column="accuracy", by=["beta", "gamma"])
        plt.title("Clean Test set accuracy")
        plt.ylabel("Test set accuracy")
        plt.show()
    else:
        print(everything)


def read_gaussian(args):

    df = pd.read_pickle("test_results/gaussian_noise.pkl")
    df = df.drop(
        ["k", "m", "seed", "sample", "epsilon","accuracy"],
        axis=1)
    pre_agg = df.groupby(["beta", "gamma"])
    everything = pre_agg.agg(["mean"])
    if args.matplotlib:
        df.boxplot(column="accuracy_gauss", by=["beta", "gamma"], sym='')
        plt.title("Test set accuracy under gaussian noise")
        plt.ylabel("Test set accuracy")
        plt.show()
    else:
        print(everything)


def read_dropout(args):

    df = pd.read_pickle("test_results/dropout.pkl")
    df = df.drop(["k", "m", "seed", "sample"], axis=1)
    pre_agg = df.groupby(["beta", "gamma", "dropout"])
    everything = pre_agg.agg(["mean"])
    if args.matplotlib:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        df.query("dropout==0.25").boxplot(
            column="accuracy_dropout", by=["beta", "gamma"], ax=ax1, sym='')
        df.query("dropout==0.40").boxplot(
            column="accuracy_dropout",
            by=["beta", "gamma"], ax=ax2, sym='')
        fig.suptitle("Test set accuracy under Dropout")
        ax1.set_ylabel("Test set accuracy")
        ax1.set_title("Dropout = 0.25")
        ax2.set_ylabel("Test set accuracy")
        ax2.set_title("Dropout = 0.40")
        plt.show()
    else:
        print(everything)


def main():

    all_results = ["clean", "gaussian", "dropout", "all"]

    parser = argparse.ArgumentParser(description='Read Training results')
    parser.add_argument(
        '--result', choices=all_results,
        default='all', help='which result to read')
    parser.add_argument(
        '--matplotlib', action='store_true',
        help='display results with matplotlib')
    args = parser.parse_args()

    if args.result == "all" or args.result == "clean":
        if args.result == "all" and not args.matplotlib:
            print("Clean set results:")
        read_clean(args)

    if args.result == "all" or args.result == "gaussian":
        if args.result == "all" and not args.matplotlib:
            print("Gaussian Noise results:")
        read_gaussian(args)

    if args.result == "all" or args.result == "dropout":
        if args.result == "all" and not args.matplotlib:
            print("Dropout results:")
        read_dropout(args)

if __name__ == "__main__":
    main()
