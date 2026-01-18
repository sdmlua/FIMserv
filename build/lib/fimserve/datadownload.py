import os
import csv
import pandas as pd
import subprocess


def setup_directories():
    parent_dir = os.getcwd()
    code_dir = os.path.join(parent_dir, "code", "inundation-mapping")
    data_dir = os.path.join(parent_dir, "data", "inputs")
    output_dir = os.path.join(parent_dir, "output")

    # Create the directories if not exist
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return code_dir, data_dir, output_dir


def clone_repository(code_dir, version=None):
    repo_path = os.path.join(code_dir)
    repo_url = "https://github.com/NOAA-OWP/inundation-mapping.git"

    # Map version strings to git tags
    version_tag_map = {
        "4.5": "v4.6.1.4",
        "default": "main",
    }

    # Determine tag to checkout
    version_tag = version_tag_map.get(version, version_tag_map["default"])
    if os.path.exists(repo_path) and os.listdir(repo_path):
        print(f"Repository already exists at {repo_path}. Skipping clone.")
        return

    # Clone repo
    subprocess.run(["git", "clone", repo_url, repo_path], check=True)

    # Checkout specific version/tag (only if not main)
    if version_tag != "main":
        try:
            subprocess.run(["git", "checkout", version_tag], cwd=repo_path, check=True)
        except subprocess.CalledProcessError:
            subprocess.run(["git", "tag"], cwd=repo_path)  # List available tags
            raise
    print(f"Repository cloned into: {repo_path} (version: {version_tag})")


def download_data(huc_number, base_dir, version=None):
    output_dir = os.path.join(base_dir, f"flood_{huc_number}", str(huc_number))
    os.makedirs(output_dir, exist_ok=True)

    # Determine S3 path based on version
    if version == "4.5":
        s3_path = f"s3://ciroh-owp-hand-fim/hand_fim_4_5_2_11/{huc_number}/"
    else:
        s3_path = f"s3://ciroh-owp-hand-fim/hand_fim_4_8_7_2/{huc_number}/"

    # Run the AWS CLI command with no-sign-request
    cmd = f"aws s3 sync {s3_path} {output_dir} --no-sign-request"
    os.system(cmd)
    print(f"Data for HUC {huc_number} downloaded to {output_dir}")

    # Copy branch_ids.csv to fim_inputs.csv
    hydrotable_path = os.path.join(output_dir, "branch_ids.csv")
    fim_inputs_path = os.path.join(base_dir, f"flood_{huc_number}", "fim_inputs.csv")

    with (
        open(hydrotable_path, "r") as infile,
        open(fim_inputs_path, "w", newline="") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            writer.writerow(row)

    print(f"Copied {hydrotable_path} to {fim_inputs_path} as fim_inputs.csv.")


def uniqueFID(hydrotable, fid_dir, stream_order=None):
    hydrotable_df = pd.read_csv(hydrotable)

    if stream_order:
        if isinstance(stream_order, str) and (
            ">=" in stream_order
            or "<=" in stream_order
            or ">" in stream_order
            or "<" in stream_order
        ):
            # Handle conditions like ">=3", "<=4", etc.
            operator, value = None, None
            if ">=" in stream_order:
                operator, value = ">=", int(stream_order.split(">=")[1])
            elif "<=" in stream_order:
                operator, value = "<=", int(stream_order.split("<=")[1])
            elif ">" in stream_order:
                operator, value = ">", int(stream_order.split(">")[1])
            elif "<" in stream_order:
                operator, value = "<", int(stream_order.split("<")[1])

            hydrotable_df["order_"] = hydrotable_df["order_"].astype(int)
            if operator == ">=":
                hydrotable_df_filtered = hydrotable_df[hydrotable_df["order_"] >= value]
            elif operator == "<=":
                hydrotable_df_filtered = hydrotable_df[hydrotable_df["order_"] <= value]
            elif operator == ">":
                hydrotable_df_filtered = hydrotable_df[hydrotable_df["order_"] > value]
            elif operator == "<":
                hydrotable_df_filtered = hydrotable_df[hydrotable_df["order_"] < value]

        elif isinstance(stream_order, (list, int)):
            if isinstance(stream_order, int):
                stream_order = [stream_order]
            stream_order = [str(order) for order in stream_order]
            hydrotable_df["order_"] = hydrotable_df["order_"].astype(str)
            hydrotable_df_filtered = hydrotable_df[
                hydrotable_df["order_"].isin(stream_order)
            ]
        else:
            raise ValueError(
                "Invalid stream_order format. Use a list of values, a single value, or a condition like '>=3'."
            )
    else:
        # No filter, use the whole dataset
        hydrotable_df_filtered = hydrotable_df

    unique_FIDs = hydrotable_df_filtered["feature_id"].drop_duplicates()
    unique_FIDs_df = pd.DataFrame(unique_FIDs, columns=["feature_id"])
    unique_FIDs_df.to_csv(fid_dir, index=False)
    print(f"Unique feature IDs saved to {fid_dir}.")


# IN the replacement of Docker, the environement variables are set in the .env file
def EnvFile(code_dir):
    env_dir = os.path.join(code_dir, ".env")
    env_content = """
    inputsDir=inputs
    outputsDir=output
    """
    os.makedirs(os.path.dirname(env_dir), exist_ok=True)

    with open(env_dir, "w") as f:
        f.write(env_content)


def DownloadHUC8(huc, stream_order=None, version=None):
    code_dir, data_dir, output_dir = setup_directories()
    clone_repository(code_dir, version)

    # if huc is not str:
    huc = str(huc)
    download_data(huc, output_dir, version)
    EnvFile(code_dir)
    HUC_dir = os.path.join(output_dir, f"flood_{huc}")
    hydrotable_dir = os.path.join(HUC_dir, str(huc), "hydrotable.csv")
    featureID_dir = os.path.join(HUC_dir, f"feature_IDs.csv")
    if stream_order is None:
        uniqueFID(hydrotable_dir, featureID_dir)
    else:
        uniqueFID(hydrotable_dir, featureID_dir, stream_order)
