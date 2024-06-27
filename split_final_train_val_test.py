import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "E:/code/Py_workplace/spacenet8/Data/train_val_csv/modis_sp8_all.csv"  # 替换为你的文件路径
data = pd.read_csv(file_path)

# Check the first few rows of the dataset
print(data.head())

random_seeds = [4353, 7845, 1297, 6184, 2134, 8967, 1023, 5348, 7621, 4976]
## for the mix dataset:
# Splitting the data into training, validation, and test sets
for seed in random_seeds:
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)

    # Output the sizes of each set
    train_size = train_data.shape[0]
    val_size = val_data.shape[0]
    test_size = test_data.shape[0]

    print(f'随机种子: {seed}')
    print(f'训练集大小: {train_size}')
    print(f'验证集大小: {val_size}')
    print(f'测试集大小: {test_size}')

    # Save the datasets to CSV files
    train_data.to_csv(f'train_data_mix_seed_{seed}.csv', index=False)
    val_data.to_csv(f'val_data_mix_seed_{seed}.csv', index=False)
    test_data.to_csv(f'test_data_mix_seed_{seed}.csv', index=False)

## for the Louisiana dataset:
filtered_data = data[data.iloc[:, 0].str.contains('Louisiana', na=False)]
# Splitting the data into training, validation, and test sets
for seed in random_seeds:
    train_data, temp_data = train_test_split(filtered_data, test_size=0.3, random_state=seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)

    # Output the sizes of each set
    train_size = train_data.shape[0]
    val_size = val_data.shape[0]
    test_size = test_data.shape[0]

    print(f'随机种子: {seed}')
    print(f'训练集大小: {train_size}')
    print(f'验证集大小: {val_size}')
    print(f'测试集大小: {test_size}')

    # Save the datasets to CSV files
    train_data.to_csv(f'train_data_lou_seed_{seed}.csv', index=False)
    val_data.to_csv(f'val_data_lou_seed_{seed}.csv', index=False)
    test_data.to_csv(f'test_data_lou_seed_{seed}.csv', index=False)
