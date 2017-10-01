import matplotlib
import pandas as pd

matplotlib.use('agg')
from os.path import join

results_root = '../results/'
plot_fn = '../plots/'

def get_pred_file(model, discounting_type, normalize):
    pred_format = "{model}_{discounting_type}_{norm}_predictions.csv"
    return join(results_root, pred_format.format(
        model=model, discounting_type=discounting_type, norm="n" if normalize else "un"))

models = ["EXP", "HY", "ITCH", "QHY"]
discounting_types = ["TIME", "PROB", "EFFORT"]
#reward_types = ["social", "health", "money"]
normalizes = [True, False]
dfs = []

for normalize in normalizes:

    for discounting_type in discounting_types:

        for model in models:
            df = pd.read_csv(get_pred_file(model, discounting_type, normalize))
            df['reward_type'] = df['key'].apply(lambda fn: fn.split('_')[-1].split('.')[0])
            df['model'] = model
            df['discounting_type'] = discounting_type
            df['normalize'] = normalize
            dfs.append(df)

df_merge = pd.concat(dfs)
df_merge.to_csv('../results/merged_predictions.csv')