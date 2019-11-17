import pandas as pd
import readline
data = pd.read_csv('mass16_2.txt')
header(data)
head(data)
data.shape
data = pd.read_csv('mass16_2.txt', sep='\t')
data.shape
print(data)
data[['N']] = data.apply(lambda row: row.A - row.Z, axis = 1)
data['N'] = data.apply(lambda row: row.A - row.Z, axis = 1)
print(data)
data['Zpar'] = data.apply(lambda row: row.Z % 2, axis = 1)
data['Npar'] = data.apply(lambda row: row.N % 2, axis = 1)
pd.set_option("display.max_rows", 10)
print(data)
data = data['A', 'EL', 'Z', 'Zpra', 'N', 'Npar', 'B/A', 'est']
data = data[['A', 'EL', 'Z', 'Zpra', 'N', 'Npar', 'B/A', 'est']]
data = data['A', 'EL', 'Z', 'Zpar', 'N', 'Npar', 'B/A', 'est']
data = data[['A', 'EL', 'Z', 'Zpar', 'N', 'Npar', 'B/A', 'est']]
print(data)
data = data[['EL', 'Z', 'Zpar', 'N', 'Npar', 'A', 'B/A', 'est']]
print(data)
pd.set_option("display.max_rows", 20)
print(data)
data = data[['A', 'EL', 'Z', 'Zpar', 'N', 'Npar', 'B/A', 'est']]
print(data)
data[2]
data[2,]
data[,2]
data[_,2]
data[2,_]
print(data)
data_train = data.loc[data['est'] == 0]
data_test = data.loc[data['est'] == 1]
print(data_train)
print(data_test)
x_train = data_train[['Z', 'Zpar', 'N', 'Npar']]
y_train = data_train[['B/A']]
x_test = data_test[['Z', 'Zpar', 'N', 'Npar']]
y_test = data_test[['B/A']]
print(x_test)
print(y_test)
y_train.max()
y_test.max()
x_train.max()
x_test.max()
y_test = y_test/8800
y_train = y_train/8800
x_train[['Z', 'N']] = x_train[['Z', 'N']] / 200.
x_test[['Z', 'N']] = x_test[['Z', 'N']] / 200.
print(x_train)
print(x_test)
readline.write_history_file('input.py')
