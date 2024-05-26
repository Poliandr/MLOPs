import os
from vega_datasets import data


# Download dataset
def load_data():
    dataset = data.cars()
    return dataset


# Save dataset to file
def save_data(folder, data, name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, f"{name}.csv")
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


# Split dataset
def main():
    print(data.list_datasets())
    dataset = load_data()
    test_dataset = dataset.iloc[-int(len(dataset) * 0.1):]

    # Save train dataset
    save_data("train",
              dataset,
              "train_data")
    # Save test dataset
    save_data("test",
              test_dataset,
              "test_data")


if __name__ == "__main__":
    main()
