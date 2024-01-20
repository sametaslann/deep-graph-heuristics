import extract_graphs
import collect_data


heuristic_names = ["afisa_original", "dsatur", "ilsts", "redls", "tabu_weight"]
collect_data.read_graph_results(heuristic_names)

instance_list = collect_data.get_instance_list()
extract_graphs.read_graphs(instance_list)
