import os
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)


def get_data():
    print(os.getcwd())
    files = os.listdir()
    for file in files:
        if 'time' in file:
            path = os.path.join(os.getcwd(), file)
            readpathfiles = os.listdir(path)
            for x in readpathfiles:
                if 'jena' in x:
                    datafile = pd.read_csv(os.path.join(path, x))
                    print(datafile.head())
                    print(datafile.columns)
                    return datafile


if __name__ == '__main__':
    data = get_data()


from sklearn.preprocessing import StandardScaler
X = data[['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',
       'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)']]
transformer = StandardScaler().fit(X)
x_standardized = transformer.transform(X)
x_standardized = pd.DataFrame(x_standardized)
x_standardized['wd (deg)'] = data['wd (deg)']
print(x_standardized.shape)
print(x_standardized.head())
array = np.array(x_standardized)


# def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
#     if max_index is None:
#         max_index = len(data) - delay - 1
#     i = min_index + lookback
#     while 1:
#         if shuffle:
#             rows = np.random.randint(
#                 min_index + lookback, max_index, size=batch_size)
#         else:
#             if i + batch_size >= max_index:
#                 i = min_index + lookback
#             rows = np.arange(i, min(i + batch_size, max_index))
#             i += len(rows)
#
#         samples = np.zeros((len(rows),
#                            lookback // step,
#                            data.shape[-1]))
#         targets = np.zeros((len(rows),))
#         for j, row in enumerate(rows):
#             indices = range(rows[j] - lookback, rows[j], step)
#             samples[j] = data[indices]
#             targets[j] = data[rows[j] + delay][1]
#         yield samples, targets
#
#
#
#
