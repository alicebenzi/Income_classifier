__author__ = 'alicebenziger'

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def label_encoder(data, binary_cols, categorical_cols):
    label_enc = LabelEncoder()
    for col in categorical_cols:
        label_enc.fit(data[col])
        data[col] = label_enc.transform(data[col])
    encoded_categorical = np.array(data[categorical_cols])

    for col in binary_cols:
        label_enc.fit(data[col])
        data[col] = label_enc.transform(data[col])
    encoded_binary = np.array(data[binary_cols])
    return encoded_categorical, encoded_binary


def dummy_encoder(train_X,test_X,categorical_variable_list):
    enc = OneHotEncoder(categorical_features=categorical_variable_list)
    train_X = enc.fit_transform(train_X).toarray()
    test_X = enc.transform(test_X).toarray()
    return train_X, test_X


def normalize(X):
    normalizer = preprocessing.Normalizer().fit(X)
    normalized_X = normalizer.transform(X)
    return normalized_X



if __name__ == "__main__":

    train = pd.read_csv('train_new.csv',header=0)
    test = pd.read_csv('test_new.csv',header=0)