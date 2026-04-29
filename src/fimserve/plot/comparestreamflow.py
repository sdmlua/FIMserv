import os
import matplotlib.pyplot as plt

from .nwmfid import getFIDdata
from .usgs import getUSGSdata
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


def plotcomparision(
    data_dir_nwm, data_dir_usgs, feature_id, usgs_site, output_dir, start_date, end_date
):
    nwm_data = getFIDdata(data_dir_nwm, feature_id, start_date, end_date)
    usgs_data = getUSGSdata(data_dir_usgs, usgs_site, start_date, end_date)

    plt.figure(figsize=(10, 5))
    plt.plot(
        nwm_data["Date"],
        nwm_data["Discharge"],
        label=f"NWM Streamflow of feature ID {feature_id}",
        linestyle="solid",
        color="#167693",
        linewidth=2,
    )
    plt.plot(
        usgs_data["Date"],
        usgs_data["Discharge"],
        label=f"USGS Streamflow of gauged site {usgs_site}",
        linestyle="dashed",
        color="#BF4037",
        linewidth=2,
    )

    plt.xlabel("Date (Hourly)", fontsize=14)
    plt.ylabel("Discharge (m³/s)", fontsize=14)
    plt.title("Discharge comparison between USGS and NWM streamflow", fontsize=16)
    plt.legend()
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid(True, which="both", linestyle="-", linewidth=0.3)

    plt_dir = os.path.join(output_dir, "Plots")
    os.makedirs(plt_dir, exist_ok=True)
    plot_filename = f"NWMvsUSGS_{usgs_site}.png"
    plot_path = os.path.join(plt_dir, plot_filename)
    plt.savefig(plot_path, dpi=500, bbox_inches="tight")
    print(f"Static plot saved to: {plot_path}")

    plt.close()

    env = _notebook_env()

    if env != "none":
        try:
            import plotly.graph_objects as go
            from IPython.display import display, HTML

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=nwm_data["Date"],
                    y=nwm_data["Discharge"],
                    mode="lines",
                    name=f"NWM Feature ID: {feature_id}",
                    line=dict(color="#167693", width=2, dash="solid"),
                    hovertemplate=(
                        "<b>NWM Feature ID:</b> " + str(feature_id) + "<br>"
                        "<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br>"
                        "<b>Discharge:</b> %{y:.4f} m³/s"
                        "<extra></extra>"
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=usgs_data["Date"],
                    y=usgs_data["Discharge"],
                    mode="lines",
                    name=f"USGS Site: {usgs_site}",
                    line=dict(color="#BF4037", width=2, dash="dash"),
                    hovertemplate=(
                        "<b>USGS Site:</b> " + str(usgs_site) + "<br>"
                        "<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br>"
                        "<b>Discharge:</b> %{y:.4f} m³/s"
                        "<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title=dict(
                    text="Discharge Comparison: USGS vs NWM Streamflow",
                    font=dict(size=18),
                ),
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


def CompareNWMnUSGSStreamflow(huc, feature_id, usgs_site, start_date, end_date):
    code_dir, data_dir, output_dir = setup_directories()
    discharge_dir_nwm = os.path.join(
        output_dir, f"flood_{huc}", "discharge", "nwm30_retrospective"
    )
    discharge_dir_usgs = os.path.join(
        output_dir, f"flood_{huc}", "discharge", "usgs_streamflow"
    )
    HUC_dir = os.path.join(output_dir, f"flood_{huc}")
    plotcomparision(
        discharge_dir_nwm,
        discharge_dir_usgs,
        feature_id,
        usgs_site,
        HUC_dir,
        start_date,
        end_date,
    )
