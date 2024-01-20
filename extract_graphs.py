import collect_data
import numpy as np
import csv
import sys

csv.field_size_limit(sys.maxsize)


def read_graphs(instance_list):
    for instance in instance_list:
        file_path = f"graph_instances/{instance}.col"
        with open(file_path, "r") as file:
            content = file.read()
            print("Reading instance", instance)
            features = parse_dimacs(content, instance)
            write_to_csv(instance, features)


def parse_dimacs(content, instance):
    lines = content.strip().split("\n")

    header = lines[1].split()
    num_vertices = int(header[2])
    num_edges = int(header[3])
    ratio = num_edges / num_vertices
    density = 2 * num_vertices / (num_edges * (num_edges - 1))

    graph = [num_vertices * [0] for _ in range(num_vertices)]
    # graph_representation = ""
    for line in lines[1:]:
        parts = line.split()
        if parts[0] == "e":
            vertex1, vertex2 = map(int, parts[1:3])
            # graph_representation += str(vertex1) + "-" + str(vertex2) + " "
            graph[vertex1 - 1][vertex2 - 1] = 1
            graph[vertex2 - 1][vertex1 - 1] = 1

    counts = [row.count(1) for row in graph]
    min_count, min_row = min((row.count(1), i) for i, row in enumerate(graph))
    max_count, max_row = max((row.count(1), i) for i, row in enumerate(graph))
    mean = np.mean(counts)
    median = np.median(counts)
    q1 = np.percentile(counts, 25)
    q3 = np.percentile(counts, 75)
    variation_coefficient = np.std(counts) / mean if mean else 0
    counts_np = np.array(counts)
    probabilities = counts_np / np.sum(counts_np) if np.sum(counts_np) else counts_np
    entropy = calculate_entropy(probabilities)

    # print(len(graph_representation))
    features = [
        num_vertices,
        num_edges,
        ratio,
        density,
        min_count,
        max_count,
        mean,
        median,
        q1,
        q3,
        variation_coefficient,
        entropy,
        # graph_representation,
    ]
    return features


def write_to_csv(instance, features):
    existing_data = []
    with open("output2.csv", "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        existing_data = list(reader)

    # create column names
    if instance == "C2000.5":
        values = [
            "num_vertices",
            "num_edges",
            "ratio",
            "density",
            "min_count",
            "max_count",
            "mean",
            "median",
            "q1",
            "q3",
            "variation_coefficient",
            "entropy",
            # "#graph_representation",
        ]
        existing_data[0].extend(values)

    for row in existing_data:
        if row[0] == instance:
            row.extend(features)
            break

    with open("output2.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(existing_data)


def calculate_entropy(counts):
    probabilities = counts / np.sum(counts) if np.sum(counts) else counts
    entropy_value = -np.sum(
        probabilities * np.log2(probabilities + np.finfo(float).eps)
    )
    return entropy_value


# instance_list = collect_data.get_instance_list()
# read_graphs(instance_list)
