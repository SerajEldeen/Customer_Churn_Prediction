def select_features(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
