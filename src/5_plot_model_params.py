import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import parallel_coordinates
import seaborn as sns
import os
from os.path import join


def preprocess(data, parameters, metadata, normalize=False):
    split_path = lambda s: os.path.split(s)[-1]
    data['subnum'] = data['key'].apply(lambda s: int(split_path(s).split('_')[0]))
    data['rewardType'] = data['key'].apply(lambda s: split_path(s).split('_')[2].split('.')[0])

    data = data.merge(metadata, left_on='subnum', right_on='Subnum')
    data = data.dropna()
    if normalize:
        for i in range(data.shape[0]):
            w = max(abs(data.loc[i, parameters]))
            data.loc[i, 'weight'] = w if w > 0 else 1

        for parameter in parameters:
            data[parameter] = data[parameter] / data['weight']

        parameters.append('weight')

    return data


results_dir = join('..', 'results')
plot_dir = join('..', 'plots')
params_dir = join(plot_dir, 'params')

id_string = "{model}_{discounting_type}_{norm}"
metrics_format = '{id_string}_metrics.csv'
model_to_parameters = {
    "ITCH": ['beta_1', 'beta_tA', 'beta_tR', 'beta_xA', 'beta_xR'],
    "HY": ['k', 'w']
}
discounting_types = ["TIME", "PROB"]
norms = [True, False]

metadata = pd.read_csv(join('..', 'data', 'subject_data.csv'))
metadata['Subnum'] = metadata['partNum']
metadata = metadata[['Subnum', 'Age', 'Sex']].drop_duplicates()
metadata.set_index(metadata['Subnum'])


def plot_metrics(model, parameters, norm, discounting_type):
    id_string_formatted = id_string.format(
        model=model, norm="n" if norm else "un", discounting_type=discounting_type)
    data = pd.read_csv(join(results_dir, metrics_format.format(
        id_string=id_string_formatted)))

    data = preprocess(data, parameters, metadata, normalize=False)

    additional_parameters = ['train_accuracy', 'test_accuracy']
    model_params = parameters + additional_parameters

    plt.figure(figsize=(5 * len(model_params), 5))
    for i, parameter in enumerate(model_params):
        plt.subplot(1, len(model_params), i + 1)
        plt.tight_layout()
        males = data[data['Sex'] == 'M']
        females = data[data['Sex'] == 'F']

        X, Y = data['Age'], data[parameter]
        size = data.shape[0]
        X = X.values.reshape((size, 1))
        Y = Y.values.reshape((size, 1))
        reg = LinearRegression().fit(X, Y)
        inputs = np.linspace(0, 90, 100).reshape((100, 1))
        plt.plot(inputs, reg.predict(inputs), 'g-')

        slope, intercept, r_value, p_value, std_err = linregress(data['Age'], data[parameter])
        print("coefficients for {} vs age:".format(parameter))
        print("slope: {}, Intercept: {}, p-value: {}".format(slope, intercept, p_value))

        plt.plot(males['Age'], males[parameter], 'b.')
        plt.plot(females['Age'], females[parameter], 'r.')
        plt.title('Effect of Age on {}'.format(parameter))
        plt.xlabel('Age')
        plt.ylabel(parameter)
    plt.savefig(join(params_dir, 'age_vs_params_{}.png'.format(id_string_formatted)))
    plt.clf()

    rewardTypes = ['health', 'money', 'social']
    plt.figure(figsize=(5 * len(model_params), 5))
    for i, parameter in enumerate(model_params):
        plt.subplot(1, len(model_params), i + 1)
        plt.tight_layout()
        data_list = [data[data['rewardType'] == rewardType][parameter] for rewardType in rewardTypes]
        plt.boxplot(data_list)
        plt.xticks(range(1, len(rewardTypes) + 1), rewardTypes)
        plt.title('Effect of RewardType on {}'.format(parameter))
        plt.xlabel('RewardType')
        plt.ylabel(parameter)
    plt.savefig(join(params_dir, 'rewardType_vs_params_{}.png'.format(id_string_formatted)))
    plt.clf()

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 1, 1)
    scatter_matrix(data[data['rewardType'] == 'health'][model_params])
    # scatter_matrix(results[results['rewardType']=='social'][parameters])
    plt.savefig(join(params_dir, 'scattermatrix_{}.png'.format(id_string_formatted)))
    plt.clf()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    parallel_coordinates(data[model_params + ['rewardType']], 'rewardType')
    plt.savefig(join(params_dir, 'parallel_coords_{}.png'.format(id_string_formatted)))
    plt.clf()

    plt.figure(figsize=(30, 30))
    plt.subplot(1, 1, 1)
    sns.set()
    sns.pairplot(data[model_params + ['rewardType', 'Age']], hue="rewardType", plot_kws={'alpha': .7})
    plt.savefig(join(params_dir, 'factor_scattermatrix_{}.png'.format(id_string_formatted)))
    plt.clf()


for discounting_type in discounting_types:
    for norm in norms:
        for model, parameters in model_to_parameters.items():
            plot_metrics(model, parameters, norm, discounting_type)
