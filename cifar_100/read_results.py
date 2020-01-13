import pandas as pd
import argparse
import matplotlib.pyplot as plt


def read_clean(args):

    df = pd.read_pickle("test_results/fgsm.pkl")
    df = df.drop(["k", "m", "seed", "snr", "accuracy_before_norm",
                  "mean_epsilon", "accuracy_after_norm"], axis=1)
    pre_agg = df.groupby(["beta", "gamma"])
    everything = pre_agg.agg(["mean"])
    if args.matplotlib:
        df.boxplot(column="accuracy", by=["beta", "gamma"])
        plt.title("Clean Test set accuracy")
        plt.ylabel("Test set accuracy")
        plt.show()
    else:
        print(everything)


def read_fgsm(args):

    df = pd.read_pickle("test_results/fgsm.pkl")
    df = df.drop(["k", "m", "seed", "snr", "mean_epsilon", "accuracy"], axis=1)
    pre_agg = df.groupby(["beta", "gamma"])
    everything = pre_agg.agg(["mean"])
    if args.matplotlib:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        df.boxplot(
            column="accuracy_after_norm", by=["beta", "gamma"], ax=ax1, sym='')
        df.boxplot(
            column="accuracy_before_norm",
            by=["beta", "gamma"], ax=ax2, sym='')
        fig.suptitle("Test set accuracy under FGSM attack")
        ax1.set_ylabel("Test set accuracy")
        ax2.set_ylabel("Test set accuracy")
        plt.show()
    else:
        print(everything)


def read_gaussian(args):

    df = pd.read_pickle("test_results/gaussian_noise.pkl")
    df = df.drop(
        ["k", "m", "seed", "sample", "epsilon", "accuracy"],
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


def main():

    all_results = ["clean", "fgsm", "gaussian", "all"]

    parser = argparse.ArgumentParser(description='Read Training results')
    parser.add_argument(
        '--result', choices=all_results,
        default='all', help='which result to read')
    parser.add_argument(
        '--matplotlib', action='store_true',
        help='display results with matplotlib')
    args = parser.parse_args()

    if args.result == "all" or args.result == "clean":
        read_clean(args)

    if args.result == "all" or args.result == "fgsm":
        read_fgsm(args)

    if args.result == "all" or args.result == "gaussian":
        read_gaussian(args)

if __name__ == "__main__":
    main()
