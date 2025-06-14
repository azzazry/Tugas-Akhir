import os

def get_paths(users):
    data_dir = f"data/{users}_users"

    result_dir = f"result/{users}_users"
    data_out = os.path.join(result_dir, "data")
    logs_dir = os.path.join(result_dir, "logs")
    vis_dir = os.path.join(result_dir, "visualizations")

    os.makedirs(data_out, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    return {
        # Input path
        "data_graph_path": os.path.join(data_dir, "data_graph.pt"),
        "data_objects_path": os.path.join(data_dir, "data_objects.pkl"),
        "metadata_path": os.path.join(data_dir, "metadata.json"),
        "user_metadata_path": os.path.join(data_dir, "user_metadata.pkl"),

        # Output path
        "model_path": os.path.join(data_out, "insider_threat_graphsage.pt"),
        "training_info_path": os.path.join(data_out, "training_info.pkl"),
        "evaluation_path": os.path.join(data_out, "evaluation_results.pkl"),
        "explanation_path": os.path.join(data_out, "graphsvx_explanations.pkl"),

        # Log & vis
        "log_file": os.path.join(logs_dir, "training_log.log"),
        "eval_log_path": os.path.join(logs_dir, "evaluation_log.log"),
        "explanation_log_path": os.path.join(logs_dir, "explanation_log.log"),
        "visualization_dir": vis_dir
    }