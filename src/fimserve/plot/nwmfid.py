import pandas as pd
import os
import matplotlib.pyplot as plt

from ..datadownload import setup_directories

def _notebook_env() -> str:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return "none"

        if "google.colab" in str(shell.__class__.__module__):
            return "colab"

        import sys
        if "google.colab" in sys.modules:
            return "colab"

        if type(shell).__name__ == "ZMQInteractiveShell":
            return "notebook"

        return "none"

    except ImportError:
        return "none"


def getFIDdata(data_dir, feature_id, start_date, end_date):
    location_id = f"nwm30-{feature_id}"

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    start_dateSTR = start_date.strftime("%Y%m%d")
    end_dateSTR = end_date.strftime("%Y%m%d")
    target_file = f"{start_dateSTR}_{end_dateSTR}.parquet"

    target_dir = os.path.join(data_dir, target_file)

    if not os.path.exists(target_dir):
        raise FileNotFoundError(
            f"No NWM data found for the date range: {target_file} in {data_dir}. "
            "Please check the date range that you downloaded for that HUC."
        )

    df = pd.read_parquet(target_dir)
    matched_rows = df[df["location_id"] == location_id]

    filtered_data = matched_rows[["value_time", "value"]].copy()
    filtered_data.rename(
        columns={"value_time": "Date", "value": "Discharge"}, inplace=True
    )

    return filtered_data


def getFeatureWithMaxDischarge(data_dir, start_date, end_date):
    start_dateSTR = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_dateSTR = pd.to_datetime(end_date).strftime("%Y%m%d")
    target_file = f"{start_dateSTR}_{end_dateSTR}.parquet"

    target_dir = os.path.join(data_dir, target_file)

    if not os.path.exists(target_dir):
        raise FileNotFoundError(
            f"No NWM data found for the date range: {target_file} in {data_dir}. "
            "Please check the date range that you downloaded for that HUC."
        )

    df = pd.read_parquet(target_dir)
    max_discharge_row = df.loc[df["value"].idxmax()]
    max_feature_id = max_discharge_row["location_id"].split("-")[1]
    return max_feature_id


def plotNWMStreamflowData(dischargedata, feature_ids, output_dir, start_date, end_date):
    # Always build & save the static matplotlib figure
    plt.figure(figsize=(10, 5))
    missing_ids = []
    plotted_ids = []
    datasets = {}

    for feature_id in feature_ids:
        try:
            data = getFIDdata(dischargedata, feature_id, start_date, end_date)
            if data.empty:
                raise ValueError(f"No data for feature ID: {feature_id}")

            plt.plot(
                data["Date"],
                data["Discharge"],
                label=f"NWM streamflow for feature ID: {feature_id}",
                linewidth=2,
            )
            plotted_ids.append(feature_id)
            datasets[feature_id] = data
        except (FileNotFoundError, ValueError):
            missing_ids.append(feature_id)

    plt.xlabel("Date (Hourly)", fontsize=14)
    plt.ylabel("Discharge (m³/s)", fontsize=14)
    plt.title("NWM hourly streamflow", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle="-", linewidth=0.3)
    plt.tight_layout()
    if plotted_ids:
        plt.legend()
        plt_dir = os.path.join(output_dir, "Plots")
        os.makedirs(plt_dir, exist_ok=True)
        plot_filename = f"NWMStreamflow_{plotted_ids[0]}.png"
        plot_path = os.path.join(plt_dir, plot_filename)
        
        plt.savefig(plot_path, dpi=500, bbox_inches="tight")
        print(f"Static plot saved to: {plot_path}")

    plt.close()

    # For Notebook- render an interactive Plotly figure in the cell
    env = _notebook_env()

    if plotted_ids and env != "none":
        try:
            import plotly.graph_objects as go
            from IPython.display import display, HTML

            fig = go.Figure()

            for feature_id, data in datasets.items():
                fig.add_trace(
                    go.Scatter(
                        x=data["Date"],
                        y=data["Discharge"],
                        mode="lines",
                        name=f"Feature ID: {feature_id}",
                        line=dict(width=2),
                        hovertemplate=(
                            "<b>Feature ID:</b> " + str(feature_id) + "<br>"
                            "<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br>"
                            "<b>Discharge:</b> %{y:.4f} m³/s"
                            "<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                title=dict(text="NWM Hourly Streamflow", font=dict(size=18)),
                xaxis=dict(
                    title="Date (Hourly)",
                    tickangle=-45,
                    showgrid=True,
                    gridcolor="rgba(200,200,200,0.3)",
                ),
                yaxis=dict(
                    title="Discharge (m³/s)",
                    showgrid=True,
                    gridcolor="rgba(200,200,200,0.3)",
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                hovermode="x unified",
                template="plotly_white",
                height=500,
            )

            if env == "colab":
                html_str = fig.to_html(
                    full_html=False,
                    include_plotlyjs="cdn",
                    config={"responsive": True},
                )
                display(HTML(html_str))
            else:
                fig.show(renderer="notebook")

        except ImportError:
            pass

    # Console messages
    if not plotted_ids:
        print(
            "\033[1m****No valid data found for any provided feature IDs. "
            "Plot not generated.****\033[0m"
        )

    if missing_ids:
        print(
            f"\033[1m****Data not found for the following NWM feature IDs: "
            f"{', '.join(map(str, missing_ids))}****\033[0m"
        )

# Main function to drive the process
def plotNWMStreamflow(huc, start_date, end_date, feature_ids=None):
    code_dir, data_dir, output_dir = setup_directories()
    huc_dir = os.path.join(output_dir, f"flood_{huc}")
    discharge_dir = os.path.join(
        output_dir, f"flood_{huc}", "discharge", "nwm30_retrospective"
    )
    if feature_ids is None or not feature_ids:
        max_feature_id = getFeatureWithMaxDischarge(discharge_dir, start_date, end_date)
        print(
            f"*****No feature_id provided. Using the feature with max discharge: {max_feature_id}******"
        )
        feature_ids = [max_feature_id]

    plotNWMStreamflowData(discharge_dir, feature_ids, huc_dir, start_date, end_date)
