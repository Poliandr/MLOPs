import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    # Load data from CSV file
    return pd.read_csv(filepath)


def preprocess_data(data):
    # StandardScaler transformation
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def save_data(data, filepath):
    # Save to CSV file
    pd.DataFrame(data).to_csv(filepath, index=False)


def process_data(dataset_path, processed_dataset_path):
    # Load data
    data = load_data(dataset_path)

    X = data[["Miles_per_Gallon", "Cylinders", "Displacement", "Horsepower", "Weight_in_lbs", "Acceleration"]]
    y = data[["Name", "Year", "Origin"]]

    # Data preprocessing
    X_scaled = preprocess_data(X)

    # Сoncatenation features with labels
    processed_data = pd.concat(
        [y.reset_index(drop=True), pd.DataFrame(X_scaled)], axis=1)

    # Save processed data
    save_data(processed_data, processed_dataset_path)

    print("Предобработка данных завершена. Обработанные данные сохранены в:",
          processed_dataset_path)


def main():
    process_data("train/train_data.csv",
                 "train/processed_train_data.csv")
    process_data("test/test_data.csv",
                 "test/processed_test_data.csv")


if __name__ == "__main__":
    main()
