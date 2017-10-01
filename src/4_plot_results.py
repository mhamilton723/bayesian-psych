import matplotlib
import pandas as pd
import numpy as np
import itertools

matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import math
import os
from os.path import join

results_dir = join('..', 'results')
plot_dir = join('..', 'plots')

plot_roc = True
roc_dir = join(plot_dir, "roc_curves")
plot_cmat = True
cmat_dir = join(plot_dir, "cmats")
plot_metrics = True
metrics_dir = join(plot_dir, "dists")
plot_dists = True
dist_dir = join(plot_dir, "metrics")

models = ["ITCH", "EXP", "HY", "QHY"]
discounting_types = ["TIME", "PROB", "EFFORT"]
reward_types = ["social", "health", "money"]
normalizes = [True]

df = pd.read_csv(join(results_dir, 'merged_predictions.csv'))
df = df[df['is_test'] == True]


####### Helper functions #########
def multi_bar_graph(ys, width=.6, colors=('r', 'b', 'g')):
    assert (all(len(y) == len(ys[0]) for y in ys))
    num_bars = len(ys)
    num_entries = len(ys[0])
    x = range(num_entries)

    bar_width = width / num_bars
    offsets = np.linspace(-width / 2, width / 2, num_bars)
    for i, y in enumerate(ys):
        plt.bar(x - offsets[i], y, width=bar_width, color=colors[i % len(colors)], align='center')


######### Plotting code ########
if plot_roc:
    print("plotting roc curves")
    # by normalize, discount type, and reward type
    for normalize in normalizes:
        fig = plt.figure(figsize=(7 * len(discounting_types), 5 * len(reward_types)))
        n = 1
        for discounting_type in discounting_types:
            for reward_type in reward_types:
                for model in models:
                    test_df = df[(df['normalize'] == normalize) &
                                 (df['discounting_type'] == discounting_type) &
                                 (df['reward_type'] == reward_type) &
                                 (df['model'] == model)]

                    fpr, tpr, thresholds = roc_curve(test_df['LL'], test_df['prob'])
                    auc_val = auc(fpr, tpr)

                    plt.subplot(len(discounting_types), len(reward_types), n)
                    plt.plot(fpr, tpr, label='{} $AUC={}$'.format(model, round(auc_val, 2)))
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title('ROC Curves {} {}'.format(discounting_type, reward_type))
                    plt.legend(loc=4)
                n += 1

        plt.savefig(join(roc_dir, "roc_curves_by_dt_by_rt_norm={}.png".format(normalize)))
        plt.clf()

    # by normalize, by discounting type
    fig = plt.figure(figsize=(4.5 * len(discounting_types), 4 * len(normalizes)))
    n = 1
    for normalize in normalizes:
        for discounting_type in discounting_types:
            for model in models:
                test_df = df[(df['normalize'] == normalize) &
                             (df['discounting_type'] == discounting_type) &
                             (df['model'] == model)]

                fpr, tpr, thresholds = roc_curve(test_df['LL'], test_df['prob'])
                auc_val = auc(fpr, tpr)

                plt.subplot(len(normalizes), len(discounting_types), n)
                plt.plot(fpr, tpr, label='{} $AUC={}$'.format(model, round(auc_val, 2)))
                plt.xlabel("False Positive Rate", fontsize=12)
                plt.ylabel("True Positive Rate", fontsize=12)
                plt.title('{}'.format(discounting_type.title()), fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                # plt.title('ROC Curves {} norm={}'.format(discounting_type, normalize))
                plt.legend(loc=4, fontsize=12)

            n += 1
    plt.tight_layout()
    plt.savefig(join(roc_dir, "roc_curve_by_dt.png"))
    plt.clf()

    # by normalize, by reward_type
    fig = plt.figure(figsize=(4.5 * len(reward_types), 4 * len(normalizes)))
    n = 1
    for normalize in normalizes:
        for reward_type in reward_types:
            for model in models:
                test_df = df[(df['normalize'] == normalize) &
                             (df['reward_type'] == reward_type) &
                             (df['model'] == model)]

                fpr, tpr, thresholds = roc_curve(test_df['LL'], test_df['prob'])
                auc_val = auc(fpr, tpr)

                plt.subplot(len(normalizes), len(reward_types), n)
                plt.plot(fpr, tpr, label='{} $AUC={}$'.format(model, round(auc_val, 2)))
                plt.xlabel("False Positive Rate", fontsize=12)
                plt.ylabel("True Positive Rate", fontsize=12)
                plt.title('{}'.format(reward_type.title()), fontsize=16)
                # plt.title('ROC Curves {} norm={}'.format(reward_type, normalize))
                plt.legend(loc=4, fontsize=12)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
            n += 1
    plt.tight_layout()
    plt.savefig(join(roc_dir, "roc_curve_by_rt.png"))
    plt.clf()

if plot_cmat:
    print("plotting cmats")
    def plot_confusion_matrix(cm, classes,
                              normalize=True,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=26)
        plt.colorbar(orientation='vertical', fraction=0.046, pad=0.02)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=26)

        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)


    n = 1
    plt.figure(figsize=(10 * len(discounting_types), 10 * len(models)))
    for discounting_type in discounting_types:

        for model in models:
            test_df = df[(df['normalize'] == True) &
                         (df['discounting_type'] == discounting_type) &
                         (df['model'] == model)]
            cmat = confusion_matrix(test_df['LL'], test_df['pred'])
            plt.subplot(len(discounting_types), len(models), n)
            plot_confusion_matrix(cmat, ['SS', 'LL'], title='{} {}'.format(discounting_type, model))
            n += 1
    plt.savefig(join(cmat_dir, "cmat_by_dt.png"))
    plt.clf()

    n = 1
    plt.figure(figsize=(10 * len(discounting_types), 10 * len(models)))
    for reward_type in reward_types:
        for model in models:
            test_df = df[(df['normalize'] == True) &
                         (df['reward_type'] == reward_type) &
                         (df['model'] == model)]
            cmat = confusion_matrix(test_df['LL'], test_df['pred'])
            plt.subplot(len(reward_types), len(models), n)
            plot_confusion_matrix(cmat, ['SS', 'LL'], title='{} {}'.format(reward_type, model))
            n += 1
    plt.savefig(join(cmat_dir, "cmat_by_rt.png"))
    plt.clf()

if plot_metrics:
    print("plotting metrics")
    plt.figure(figsize=(7, 5))
    for normalize in normalizes:
        for reward_type in reward_types:
            aucs = []
            for model in models:
                test_df = df[(df['normalize'] == normalize) &
                             (df['reward_type'] == reward_type) &
                             (df['model'] == model)]

                fpr, tpr, thresholds = roc_curve(test_df['LL'], test_df['prob'])
                auc_val = auc(fpr, tpr)
                aucs.append(auc_val)
            plt.plot(aucs, 'o', ms=15, label=reward_type, alpha=.7)
        plt.ylabel('AUC')
        plt.xlabel('Model Type')
        plt.xlim(-.1, len(models) - 1 + .1)
        plt.xticks(range(len(models)), models)
        plt.legend(loc="upper left")
        plt.savefig(join(metrics_dir, "AUC_vs_rt_norm={}.png".format(normalize)))
        plt.clf()


    def accuracy(x, y):
        accuracy = (x == y).mean()
        error = (x == y).std() / math.sqrt(len(x))
        return accuracy, error


    plt.figure(figsize=(6, 5))
    for normalize in normalizes:
        ax = plt.subplot(1, 1, 1)
        for reward_type in reward_types:
            accs, confs = [], []
            for model in models:
                test_df = df[(df['normalize'] == normalize) &
                             (df['reward_type'] == reward_type) &
                             (df['model'] == model)]
                acc, err = accuracy(test_df['LL'], test_df['pred'])
                accs.append(acc)
                confs.append(2 * err)
            plt.errorbar(range(len(accs)), accs, fmt='o', yerr=confs, ms=15, label=reward_type.title(), alpha=.7)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Model Type', fontsize=14)
        plt.xlim(-.1, len(models) - 1 + .1)
        plt.xticks(range(len(models)), models, fontsize=12)
        plt.title('Accuracy by Reward Type', fontsize=16)
        plt.yticks(fontsize=12)
        ax.legend(loc='lower left', fontsize=12)
        plt.savefig(join(metrics_dir, "Accuracy_vs_rt_norm={}.png".format(normalize)))
        plt.clf()

    plt.figure(figsize=(6, 5))
    for normalize in normalizes:
        ax = plt.subplot(1, 1, 1)
        for discounting_type in discounting_types:
            accs, confs = [], []
            for model in models:
                test_df = df[(df['normalize'] == normalize) &
                             (df['discounting_type'] == discounting_type) &
                             (df['model'] == model)]
                acc, err = accuracy(test_df['LL'], test_df['pred'])
                accs.append(acc)
                confs.append(2 * err)
            plt.errorbar(range(len(accs)), accs, fmt='o', yerr=confs, ms=15, label=discounting_type.title(), alpha=.7)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Model Type', fontsize=14)
        plt.xlim(-.1, len(models) - 1 + .1)
        plt.xticks(range(len(models)), models, fontsize=12)
        plt.title('Accuracy by Dicsount Type', fontsize=16)
        plt.yticks(fontsize=12)
        ax.legend(loc='lower left', fontsize=12)
        plt.savefig(join(metrics_dir, "Accuracy_vs_dt_norm={}.png".format(normalize)))
        plt.clf()

    plt.figure(figsize=(8 * 3, 5))
    for normalize in normalizes:
        n = 1
        for discounting_type in discounting_types:
            ax = plt.subplot(1, 3, n)
            for reward_type in reward_types:
                accs, confs = [], []
                for model in models:
                    test_df = df[(df['normalize'] == normalize) &
                                 (df['reward_type'] == reward_type) &
                                 (df['discounting_type'] == discounting_type) &
                                 (df['model'] == model)]
                    acc, err = accuracy(test_df['LL'], test_df['pred'])
                    accs.append(acc)
                    confs.append(2 * err)
                plt.errorbar(range(len(accs)), accs, fmt='o', yerr=confs, ms=15, label=reward_type, alpha=.7)
            plt.ylabel('Accuracy')
            plt.xlabel('Model Type')
            plt.title(discounting_type)
            plt.xlim(-.1, len(models) - 1 + .1)
            plt.ylim(.3, 1.0)
            plt.xticks(range(len(models)), models)

            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            n += 1
        plt.savefig(join(metrics_dir, "Accuracy_vs_dt_rt_norm={}.png".format(normalize)))
        plt.clf()

if plot_dists:
    print("plotting distributions")
    grouped_means = df.groupby(['discounting_type', 'reward_type'])['LL'].mean()
    print(grouped_means)
    plt.figure(figsize=(6, 5))
    for discounting_type in discounting_types:
        x = np.array(grouped_means[discounting_type])
        x_ticks = grouped_means[discounting_type].index
        plt.plot(x, 'o', label=discounting_type, ms=9)
        plt.ylabel('Probability of LL Choices')
        plt.xlabel(reward_type)
        plt.title("Fraction of LL Choices")
        plt.xlim(-.1, len(x_ticks) - 1 + .1)
        plt.ylim(0, 1.0)
        plt.xticks(range(len(x_ticks)), x_ticks)
        plt.legend(loc="lower left")
    plt.savefig(join(dist_dir, "LL_means.png"))
