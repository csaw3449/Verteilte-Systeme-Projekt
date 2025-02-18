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
    averages_latency_all = {}
    averages_full_time = {}

    for dataset_name, dataset in datasets.items():
        averages = {
            "edge before cloud": (dataset["edge_end1"] - dataset["edge_start1"]).mean(),
            "cloud": (dataset["cloud_end"] - dataset["cloud_start"]).mean(),
            "edge after cloud": (dataset["edge_end2"] - dataset["edge_start2"]).mean()
        }
        averages_layers[dataset_name] = averages

        averages_latency = {
            "iot-egde": (dataset["edge_start1"] - dataset["iot_start"]).mean(),
            "edge-cloud": (dataset["cloud_start"] - dataset["edge_end1"]).mean(),
            "cloud-edge": (dataset["edge_start2"] - dataset["cloud_end"]).mean(),
            "edge-iot": (dataset["iot_end"] - dataset["edge_end2"]).mean()
        }
        averages_latency_all[dataset_name] = averages_latency

        averages_full_time[dataset_name] = (dataset["iot_end"] - dataset["iot_start"]).mean()

    '''create bar chart of layers'''
    df_layers = pd.DataFrame(averages_layers)
    df_layers.plot(kind="bar")
    plt.title("Average Durations of Layers")
    plt.ylabel("Duration in seconds")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("./graphs/averages_layers.png")
    plt.clf()

    '''create bar chart of latency'''
    df_latency = pd.DataFrame(averages_latency_all)
    df_latency.plot(kind="bar")
    plt.title("Average Latency of Layers")
    plt.ylabel("Duration in seconds")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("./graphs/averages_latency_all.png")
    plt.clf()

    '''create bar chart of full time'''
    df_full_time = pd.DataFrame(averages_full_time, index=["full time"])
    df_full_time.plot(kind="bar")
    plt.title("Average Full Time")
    plt.ylabel("Duration in seconds")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("./graphs/averages_full_time.png")
    plt.clf()

    print("Created bar_charts for datasets")


def combined_latency_histograms(datasets, colors):
    """Create one histogram per latency type with combined data."""
    latency_calculations = {
        "IoT to Edge 1": ("iot_start", "edge_start1"),
        "Edge 1 Processing": ("edge_start1", "edge_end1"),
        "Edge 1 to Cloud": ("edge_end1", "cloud_start"),
        "Cloud Processing": ("cloud_start", "cloud_end"),
        "Cloud to Edge 2": ("cloud_end", "edge_start2"),
        "Edge 2 Processing": ("edge_start2", "edge_end2"),
        "Edge 2 to IoT": ("edge_end2", "iot_end"),
    }

    for latency_label, (start_col, end_col) in latency_calculations.items():
        plt.figure(figsize=(10, 6))
        for (dataset_name, dataset), color in zip(datasets.items(), colors):
            try:
                if start_col in dataset.columns and end_col in dataset.columns:
                    latencies = dataset[end_col] - dataset[start_col]
                    plt.hist(latencies, bins=20, alpha=0.5, color=color, label=dataset_name, edgecolor='black')
            except KeyError as e:
                print(f"Missing data for {latency_label} in {dataset_name}: {e}")
        plt.title(f"Latency Histogram: {latency_label}")
        plt.xlabel("Latency (seconds)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./graphs/combined_histogram_{latency_label.replace(' ', '_')}.png")
        plt.clf()

def main():
    datasets = {}
    base_path = "./times"
    colors = ["blue", "green", "red", "purple", "orange"]  # Different colors for datasets
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            datasets[folder] = process_folder(folder_path)
            print(f"Processed folder {folder}")
    combined_latency_histograms(datasets, colors)
    print("Combined histograms created.")



if __name__ == "__main__":
    main()