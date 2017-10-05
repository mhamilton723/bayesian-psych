from collections import defaultdict
from os.path import join
import pandas as pd
from models import Hyperbolic, ITCH, Exponential, QuasiHyperbolic
import numpy as np

data_path = join('../', 'data')
results_path = join('../', 'results')


def accuracy(df):
    return (df['pred'] == df['LL']).mean()


def run_experiments(discount_type, model_name, normalize):
    model_name_to_constructor = {"HY": Hyperbolic,
                                 "ITCH": ITCH,
                                 "EXP": Exponential,
                                 "QHY": QuasiHyperbolic}
    model_constructor = model_name_to_constructor[model_name]

    norm_name = "n" if normalize else "un"
    kwargs = {}
    if normalize:
        kwargs['x1_col'] = 'x1_n'
        kwargs['x2_col'] = 'x2_n'

    if normalize and discount_type == "TIME":
        kwargs['t1_col'] = 't1_n'
        kwargs['t2_col'] = 't2_n'
    elif not normalize and discount_type == "TIME":
        kwargs['t1_col'] = 't1'
        kwargs['t2_col'] = 't2'
    elif discount_type == "PROB":
        kwargs['t1_col'] = '1-p1'
        kwargs['t2_col'] = '1-p2'
    elif discount_type == "EFFORT":
        kwargs['t1_col'] = 'e1'
        kwargs['t2_col'] = 'e2'
    else:
        raise ValueError("normalize={} and discount_type={} not supported".format(normalize, discount_type))

    # Load data
    data_file = join(data_path, "preprocessed_data.csv".format(discount_type))
    merged_df = pd.read_csv(data_file)
    merged_df = merged_df[merged_df['discount_type'] == discount_type]

    prediction_dfs = []
    metrics = defaultdict(list)
    num_datasets = merged_df['df_num'].max()
    for j in range(num_datasets):
        df = merged_df[merged_df['df_num'] == j]
        df_train, df_test = df[df['is_test'] == False], df[df['is_test'] == True]

        model = model_constructor(df_train, **kwargs)
        params = model.fit_MAP()

        print("experiment {}, params {}".format(j, params))

        # Record all data for roc curves
        def get_prob_df(df):
            probs = model.transform_prob(df)  # type: pd.DataFrame
            preds = (probs > .5).rename(columns={'prob': 'pred'})
            return pd.concat(
                [df[['key', 'LL', 'is_test']].reset_index(drop=True),
                 probs, preds], axis=1)

        prob_train = get_prob_df(df_train)  # type: pd.DataFrame
        prob_test = get_prob_df(df_test)  # type: pd.DataFrame
        prediction_dfs.append(pd.concat([prob_train, prob_test]))

        # record metrics
        metrics['key'].append(df['key'].iloc[0])
        for param, value in model.MAP.items():
            metrics[param].append(value)
        train_accuracy, test_accuracy = accuracy(prob_train), accuracy(prob_test)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['test_accuracy'].append(test_accuracy)
        print("train accuracy: {} test accuracy: {}".format(train_accuracy, test_accuracy))

    merged_prediction_df = pd.concat(prediction_dfs)  # type: pd.DataFrame
    metrics_df = pd.DataFrame(metrics)  # type: pd.DataFrame

    merged_prediction_df.to_csv(
        join(results_path, "{}_{}_{}_predictions.csv".format(model_name, discount_type, norm_name)))
    metrics_df.to_csv(
        join(results_path, "{}_{}_{}_metrics.csv".format(model_name, discount_type, norm_name)))

    print('Test accuracy: {}'.format(metrics_df['test_accuracy'].mean()))
    print('Training accuracy: {}'.format(metrics_df['train_accuracy'].mean()))

model_names = ["HY", "ITCH", "EXP", "QHY"]
discount_types = ["TIME", "PROB", "EFFORT"]
normalize_settings = [True, False]

for model_name in model_names:
    for discount_type in discount_types:
        for normalize in normalize_settings:
            print("model_name={} discount_type={} normalize={}".format(
                model_name, discount_type, normalize))
            run_experiments(discount_type, model_name, normalize)
