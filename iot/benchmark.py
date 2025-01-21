import os
import pandas as pd
import matplotlib.pyplot as plt

def process_folder(folder_path):
    """Reads all CSV files in one folder, treats them as a single dataset, then computes average durations."""
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]
    df_list = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        temp_df = pd.read_csv(file_path, header=0, names=[
            "iot_start", "iot_end", "edge_start1", "edge_end1",
            "cloud_start", "cloud_end", "edge_start2", "edge_end2"
        ], dtype=float) 
        df_list.append(temp_df)
    if not df_list:
        return {}
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df 

def bar_chart_of_metrics(datasets : dict):
    if not os.path.exists("./graphs"):
        os.makedirs("./graphs")

    '''compute averages of full process, edge1 and edge2 and cloud'''
    averages_layers = {}

    for dataset_name, dataset in datasets.items():
        averages = {
            "edge1": (dataset["edge_end1"].mean() - dataset["edge_start1"]).mean(),
            "cloud": (dataset["cloud_end"].mean() - dataset["cloud_start"]).mean(),
            "edge2": (dataset["edge_end2"].mean() - dataset["edge_start2"]).mean()
        }
        averages_layers[dataset_name] = averages
        plt.bar(averages.keys(), averages.values())
        plt.title(f"Average Durations of {dataset_name}")
        plt.ylabel("Duration in seconds")
        plt.savefig(f"./graphs/{dataset_name}.png")
        plt.clf()

    '''create combined bar chart'''
    df = pd.DataFrame(averages_layers)
    df.plot(kind='bar')
    plt.title("Average Durations of all datasets")
    plt.ylabel("Duration in seconds")
    plt.savefig(f"./graphs/combined.png")
    

    print("Created bar_charts for each dataset in the graphs folder")


def main():
    results = {}
    base_path = "./times"
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            results[folder] = process_folder(folder_path)
            print(f"Processed folder {folder}")
    bar_chart_of_metrics(results)


if __name__ == "__main__":
    main()