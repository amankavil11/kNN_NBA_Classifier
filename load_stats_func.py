import pandas as pd

def load_NBA_stats(return_X_y=False, target_col=None, split_test=False, split_test_elements=None):
    df_NBA_Stats = pd.read_csv('NBA_Player_Stats.csv')

    if return_X_y:
        if target_col is None:
            raise Exception('Target column along which axis to split was not provided.')
        df_X, s_y = df_NBA_Stats.drop([target_col],axis=1), df_NBA_Stats[target_col].reset_index(drop=True)
        df_NBA_Stats = (df_X, s_y)

    if split_test:
        splits = {'train':[], 'test':[]}
        if not return_X_y:
            df_NBA_Stats = (df_NBA_Stats,)
        for vectors in df_NBA_Stats:
            df_train, df_test = vectors.iloc[:-1*split_test_elements], vectors.iloc[-1*split_test_elements:].reset_index(drop=True)
            splits['train'].append(df_train)
            splits['test'].append(df_test)
        
        df_NBA_Stats = tuple(splits['train'] + splits['test'])

    return df_NBA_Stats
