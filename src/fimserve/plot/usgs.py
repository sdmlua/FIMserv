import os
import teehr
from pathlib import Path
import pandas as pd
from datetime import datetime
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


def getUSGSdata(data_dir, usgs_site, start_date, end_date):
    location_id = f"usgs-{usgs_site}"

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    start_dateSTR = start_date.strftime("%Y-%m-%d")
    end_dateSTR = end_date.strftime("%Y-%m-%d")
    target_file = f"{start_dateSTR}_{end_dateSTR}.parquet"

    target_dir = os.path.join(data_dir, target_file)

    if not os.path.exists(target_dir):
        return None

    df = pd.read_parquet(target_dir)
    matched_rows = df[df["location_id"] == location_id]

    filtered_data = matched_rows[["value_time", "value"]].copy()
    filtered_data.rename(
        columns={"value_time": "Date", "value": "Discharge"}, inplace=True
    )
    return filtered_data if not filtered_data.empty else None


def plotUSGSStreamflowData(dischargedata, usgs_sites, output_dir, start_date, end_date):
    plt.figure(figsize=(10, 5))
    missing_sites = []
    plotted_sites = []
    datasets = {}

    for usgs_site in usgs_sites:
        data = getUSGSdata(dischargedata, usgs_site, start_date, end_date)
        if data is None:
            missing_sites.append(usgs_site)
            continue

        plt.plot(
            data["Date"],
            data["Discharge"],
            label=f"USGS streamflow for gauge site: {usgs_site}",
            linewidth=2,
        )
        plotted_sites.append(usgs_site)
        datasets[usgs_site] = data

    plt.xlabel("Date (Hourly)", fontsize=14)
    plt.ylabel("Discharge (m³/s)", fontsize=14)
    plt.title("USGS hourly streamflow", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle="-", linewidth=0.3)
    plt.tight_layout()

    if plotted_sites:
        plt.legend()
        plt_dir = os.path.join(output_dir, "Plots")
        os.makedirs(plt_dir, exist_ok=True)
        plot_filename = f"USGSStreamflow_{plotted_sites[0]}.png"
        plot_path = os.path.join(plt_dir, plot_filename)
        plt.savefig(plot_path, dpi=500, bbox_inches="tight")
        print(f"Static plot saved to: {plot_path}")

    plt.close()

    env = _notebook_env()

    if plotted_sites and env != "none":
        try:
            import plotly.graph_objects as go
            from IPython.display import display, HTML

            fig = go.Figure()

            for usgs_site, data in datasets.items():
                fig.add_trace(
                    go.Scatter(
                        x=data["Date"],
                        y=data["Discharge"],
                        mode="lines",
                        name=f"USGS Site: {usgs_site}",
                        line=dict(width=2),
                        hovertemplate=(
                            "<b>USGS Site:</b> " + str(usgs_site) + "<br>"
                            "<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br>"
                            "<b>Discharge:</b> %{y:.4f} m³/s"
                            "<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                title=dict(text="USGS Hourly Streamflow", font=dict(size=18)),
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

    if missing_sites:
        print(
            f"\033[1m****Data not found for the following USGS gauge sites: "
            f"{', '.join(missing_sites)}****\033[0m"
        )


def plotUSGSStreamflow(huc, usgs_sites, start_date, end_date):
    code_dir, data_dir, output_dir = setup_directories()
    discharge_dir = os.path.join(
        output_dir, f"flood_{huc}", "discharge", "usgs_streamflow"
    )
    HUC_dir = os.path.join(output_dir, f"flood_{huc}")
    plotUSGSStreamflowData(discharge_dir, usgs_sites, HUC_dir, start_date, end_date)
