from os.path import join

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def parse_time(date):
    if date == 'Today':
        return 0.
    else:
        num_txt, interval = date.split(' ')
        num_txt2num = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8,
                       'Nine': 9}
        interval2num = {'Month': 30, 'Months': 30, 'Years': 365, 'Year': 365, 'Day': 1, 'Days': 1, 'Weeks': 7,
                        'Week': 7}
        return float(num_txt2num[num_txt] * interval2num[interval])


def parse_prob(prob):
    return float(prob.replace("%", "")) / 100.


def preprocess_data(data, discounting_type):
    data['LL'] = data['Choices.RESP'] == 8
    data['x1'] = data['magL']
    data['x2'] = data['magR']
    if discounting_type == "TIME":
        data['t1'] = data['LevelL'].apply(parse_time)
        data['t2'] = data['LevelR'].apply(parse_time)
        return data[['x1', 'x2', 't1', 't2', 'LL']]
    elif discounting_type == "PROB":
        data['1-p1'] = 1 - data['LevelL'].apply(parse_prob)
        data['1-p2'] = 1 - data['LevelR'].apply(parse_prob)
        return data[['x1', 'x2', '1-p1', '1-p2', 'LL']]
    elif discounting_type == "EFFORT":
        data['e1'] = data['cost1']
        data['e2'] = data['cost2']
        return data[['x1', 'x2', 'e1', 'e2', 'LL']]
    else:
        raise ValueError("discounting type {} not supported".format(discounting_type))


data_root = "../data"
raw_df = pd.read_csv(join(data_root, 'raw_data.csv'))
raw_df = raw_df[
    ['partNum', 'domainTask', 'domainReward', 'Choices.RESP', 'magL', 'magR', 'LevelL', 'LevelR', 'cost1', 'cost2']]

discount_types = ['TIME', 'PROB', 'EFFORT']
reward_types = ['health', 'money', 'social']
cols_to_normalize_by_dt = {'TIME': ['t1', 't2']}
cols_to_normalize_by_rt = {'health': ['x1', 'x2'], 'money': ['x1', 'x2'], 'social': ['x1', 'x2']}

dfs = []
for discount_type in discount_types:
    df = raw_df[raw_df['domainTask'] == discount_type].copy(deep=True)
    df_preproc = preprocess_data(df, discount_type).copy(deep=True)
    df_preproc['discount_type'] = df['domainTask']
    df_preproc['reward_type'] = df['domainReward']
    df_preproc['participant'] = df['partNum']
    df_preproc['key'] = df_preproc[['participant', 'discount_type', 'reward_type']] \
        .apply(lambda t: "_".join(str(entry) for entry in t), axis=1)
    df_preproc = df_preproc.sort_values(['key'])

    # Assign numbers to the different datasets for querying later
    df_preproc['df_num'] = 0
    df_num = 0
    prev_key = None
    key_col = df_preproc.columns.get_loc('key')
    df_num_col = df_preproc.columns.get_loc('df_num')
    for i in range(df_preproc.shape[0]):
        new_key = df_preproc.iloc[i, key_col]
        if prev_key is None or prev_key == new_key:
            prev_key = new_key
        else:
            prev_key = new_key
            df_num += 1
        df_preproc.iloc[i, df_num_col] = df_num

    train_i, test_i = train_test_split(range(df_preproc.shape[0]), test_size=.1, random_state=0)
    df_preproc["is_test"] = [False] * df_preproc.shape[0]
    df_preproc.iloc[test_i, df_preproc.columns.get_loc("is_test")] = True

    print(df_preproc.shape)
    dfs.append(df_preproc)

merged_df = pd.concat(dfs)

merged_df = merged_df.set_index(pd.Series(range(merged_df.shape[0])))  # type: pd.DataFrame
print(merged_df.shape)

print("merged df cols = {}".format(merged_df.columns.values))

for discount_type in discount_types:
    merged_df_subset = merged_df[merged_df['discount_type'] == discount_type]
    indicies = merged_df_subset.index
    if discount_type in cols_to_normalize_by_dt:
        cols_to_normalize = cols_to_normalize_by_dt[discount_type]
        vals_to_norm = pd.concat([merged_df_subset[col] for col in cols_to_normalize], axis=0)  # type: pd.DataFrame
        scaler = MinMaxScaler().fit(vals_to_norm.values.reshape(-1, 1))
        for col in cols_to_normalize:
            scaled_data = scaler.transform(merged_df_subset[col].values.reshape(-1, 1))
            col_name = col + "_n"
            if col_name not in merged_df.columns.values:
                merged_df[col_name] = [None] * merged_df.shape[0]
            merged_df.update(pd.DataFrame(scaled_data, index=indicies, columns=[col_name]))

for reward_type in reward_types:
    merged_df_subset = merged_df[merged_df['reward_type'] == reward_type]
    indicies = merged_df_subset.index
    if reward_type in cols_to_normalize_by_rt:
        cols_to_normalize = cols_to_normalize_by_rt[reward_type]
        vals_to_norm = pd.concat([merged_df_subset[col] for col in cols_to_normalize], axis=0)  # type: pd.DataFrame
        scaler = MinMaxScaler().fit(vals_to_norm.values.reshape(-1, 1))
        for col in cols_to_normalize:
            scaled_data = scaler.transform(merged_df_subset[col].values.reshape(-1, 1))
            col_name = col + "_n"
            if col_name not in merged_df.columns.values:
                merged_df[col_name] = [None] * merged_df.shape[0]
            merged_df.update(pd.DataFrame(scaled_data, index=indicies, columns=[col_name]))

print("merged df cols = {}".format(merged_df.columns.values))
merged_df.to_csv(join(data_root, "preprocessed_data.csv"))
