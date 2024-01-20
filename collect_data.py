import os
import csv


def get_instance_list():
    with open("instance_list_wvcp.txt", "r") as file:
        instances = file.read()
        return instances.split("\n")


def read_csv_column(file_path, column_name):
    try:
        with open(file_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            if column_name not in reader.fieldnames:
                raise ValueError(f"Column '{column_name}' not found in the CSV file.")

            last_row_result = None
            solution = None
            last_row_result_buff = None
            solution_buff = None

            for row in reader:
                last_row_result = last_row_result_buff
                solution = solution_buff
                last_row_result_buff = row[column_name]
                solution_buff = row["target"]

            return last_row_result, solution
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_csv_files(folder_path, instance_list):
    best_scores = {}
    solutions = {}
    try:
        for file_name in instance_list:
            scores = []
            for i in range(20):
                file_path = folder_path + "/" + file_name + "_" + str(i) + ".csv"
                last_row_result, solution = read_csv_column(file_path, "instance")

                if last_row_result is not None:
                    if last_row_result:
                        scores.append((last_row_result, solution))
                    else:
                        print("Column is empty.")

            # best_scores[file_name] = min(scores, key=lambda score: score[0])

            if scores:
                best_scores[file_name], solutions[file_name] = min(
                    scores, key=lambda score: score[0]
                )

            else:
                print(f"No scores for {file_name}.")
        return best_scores, solutions
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_graph_results(heuristic_names):
    csv_file_name = "output2.csv"
    instance_list = get_instance_list()

    # Writing to CSV file
    with open(csv_file_name, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        csv_writer.writerow(
            ["Instances"] + heuristic_names + ["Best heuristic"] + ["Results"]
        )

        afisa_scores, afisa_solution = read_csv_files(
            "heuristics/afisa_original", instance_list
        )
        print("1/5")
        dsatur_scores, dsatur_solution = read_csv_files(
            "heuristics/dsatur", instance_list
        )
        print("2/5")
        ilsts_scores, ilsts_solution = read_csv_files("heuristics/ilsts", instance_list)
        print("3/5")
        redls_scores, redls_solution = read_csv_files("heuristics/redls", instance_list)
        print("4/5")
        tabu_weight_scores, tabu_weight_solution = read_csv_files(
            "heuristics/tabu_weight", instance_list
        )
        print("5/5")

        for instance in instance_list:
            heuristics_and_scores = [
                ("afisa", int(afisa_scores[instance]), afisa_solution[instance]),
                ("dsatur", int(dsatur_scores[instance]), dsatur_solution[instance]),
                ("ilsts", int(ilsts_scores[instance]), ilsts_solution[instance]),
                ("redls", int(redls_scores[instance]), redls_solution[instance]),
                (
                    "tabu_weight",
                    int(tabu_weight_scores[instance]),
                    tabu_weight_solution[instance],
                ),
            ]

            best_heuristic_tuple = min(heuristics_and_scores, key=lambda x: x[1])

            best_heuristic_names = ""
            for heuristic in heuristics_and_scores:
                if heuristic[1] == best_heuristic_tuple[1]:
                    best_heuristic_names += heuristic[0] + ","

            print(best_heuristic_names)
            # best_heuristic_name = best_heuristic_tuple[0]
            best_heuristic_score = best_heuristic_tuple[1]
            best_heuristic_solution = best_heuristic_tuple[2]
            csv_writer.writerow(
                [
                    instance,
                    afisa_scores[instance],
                    dsatur_scores[instance],
                    ilsts_scores[instance],
                    redls_scores[instance],
                    tabu_weight_scores[instance],
                    best_heuristic_names,
                    best_heuristic_solution,
                ]
            )
