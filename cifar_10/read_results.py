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
        print(pre_agg.quantile([0.25, 0.5, 0.75]))


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
        print(pre_agg.quantile([0.25, 0.5, 0.75]))


def read_gaussian(args):

    df = pd.read_pickle("test_results/gaussian_noise.pkl")
    df = df.drop(
        ["k", "m", "seed", "sample", "epsilon", "accuracy", "mean_snr"],
        axis=1)
    pre_agg = df.groupby(["beta", "gamma"])
    everything = pre_agg.agg(["mean"])
    if args.matplotlib:
        df.boxplot(column="accuracy_gauss", by=["beta", "gamma"], sym='')
        plt.title("Test set accuracy under gaussian noise")
        plt.ylabel("Test set accuracy")
        plt.show()
    else:
        print(pre_agg.quantile([0.25, 0.5, 0.75]))


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
        print(pre_agg.quantile([0.25, 0.5, 0.75]).query("dropout==0.25"))
        print(pre_agg.quantile([0.25, 0.5, 0.75]).query("dropout==0.4"))


def read_pgd(args):

    df = pd.read_pickle("test_results/pgd_deepfool.pkl")
    df = df.drop(
        ["k", "m", "seed", "mean_l2", "accuracy"],
        axis=1)
    pre_agg = df.groupby(["beta", "gamma","attack"])
    everything = pre_agg.agg(["median"])
    if args.matplotlib:
        df.boxplot(column="accuracy_adversarial", by=["beta", "gamma"], sym='')
        plt.title("Test set accuracy under PGD noise")
        plt.ylabel("Test set accuracy")
        plt.show()
    else:
        print(everything)


def read_deepfool(args):

    df = pd.read_pickle("test_results/pgd_deepfool.pkl")
    df = df.drop(
        ["k", "m", "seed", "accuracy_adversarial", "accuracy"],
        axis=1)
    pre_agg = df.groupby(["beta", "gamma"])
    everything = pre_agg.agg(["mean"])
    if args.matplotlib:
        df.boxplot(column="mean_l2", by=["beta", "gamma"], sym='')
        plt.title("Test set accuracy under PGD noise")
        plt.ylabel("Test set accuracy")
        plt.show()
    else:
        print(pre_agg.quantile([0.25, 0.5, 0.75]))


def read_quantized(args):

    df = pd.read_pickle("test_results/quantized.pkl")
    df = df.drop(
        ["k", "m", "seed"],
        axis=1)
    pre_agg = df.groupby(["beta", "gamma"])
    everything = pre_agg.agg(["mean"])
    if args.matplotlib:
        df.boxplot(column="accuracy_quantized", by=["beta", "gamma"], sym='')
        plt.title("Test set accuracy under PGD noise")
        plt.ylabel("Test set accuracy")
        plt.show()
    else:
        print(pre_agg.quantile([0.25, 0.5, 0.75]))


def main():

    all_results = ["clean", "fgsm", "gaussian",
                   "dropout", "pgd", "deepfool", "quantized", "all"]

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

    if args.result == "all" or args.result == "fgsm":
        if args.result == "all" and not args.matplotlib:
            print("FGSM results:")
        read_fgsm(args)

    if args.result == "all" or args.result == "gaussian":
        if args.result == "all" and not args.matplotlib:
            print("Gaussian Noise results:")
        read_gaussian(args)

    if args.result == "all" or args.result == "dropout":
        if args.result == "all" and not args.matplotlib:
            print("Dropout results:")
        read_dropout(args)

    if args.result == "all" or args.result == "pgd":
        if args.result == "all" and not args.matplotlib:
            print("PGD results:")
        read_pgd(args)

    if args.result == "all" or args.result == "deepfool":
        if args.result == "all" and not args.matplotlib:
            print("Deepfool results:")
        read_deepfool(args)

    if args.result == "all" or args.result == "quantized":
        if args.result == "all" and not args.matplotlib:
            print("Quantized weights results:")
        read_quantized(args)


if __name__ == "__main__":
    main()
