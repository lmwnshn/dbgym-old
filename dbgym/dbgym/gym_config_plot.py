import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from dbgym.config import Config

# From Matt, figsize is (3.5,2) for half-page and (7,2) for full-page.
figsize_full = (7.0, 2.0)
figsize_half = (3.5, 2.0)
figsize_quarter = (1.75, 1.0)
fig_dpi = 600
font_mini, font_tiny, font_small, font_medium, font_large, font_huge = 4, 6, 8, 10, 12, 14

plt.rcParams["font.family"] = "Liberation Sans"
# matplotlib defaults to Type 3 fonts, which are full PostScript fonts.
# Some publishers only accept Type 42 fonts, which are PostScript wrappers around TrueType fonts.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files


class PlotConfig:
    @staticmethod
    def default_settings():
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.axisbelow"] = True
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["grid.color"] = "silver"
        plt.rcParams["grid.linestyle"] = "dotted"
        plt.rcParams["image.cmap"] = "tab10"
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["savefig.bbox"] = "tight"
        # If 0, the axes spines are eaten. https://github.com/matplotlib/matplotlib/issues/7806
        # Use pdfcrop to fix post-render.
        plt.rcParams["savefig.pad_inches"] = 0.05

    @staticmethod
    def fig_quarter():
        plt.rcParams["figure.figsize"] = figsize_quarter
        plt.rcParams["figure.dpi"] = fig_dpi
        plt.rcParams["font.size"] = font_mini
        plt.rcParams["axes.titlesize"] = font_mini
        plt.rcParams["axes.labelsize"] = font_tiny
        plt.rcParams["xtick.labelsize"] = font_mini
        plt.rcParams["ytick.labelsize"] = font_mini
        plt.rcParams["legend.fontsize"] = font_mini
        plt.rcParams["figure.titlesize"] = font_small

        for var in ["xtick", "ytick"]:
            plt.rcParams[f"{var}.major.size"] = 3.5 / 2
            plt.rcParams[f"{var}.minor.size"] = 2 / 2
            plt.rcParams[f"{var}.major.width"] = 0.8 / 2
            plt.rcParams[f"{var}.minor.width"] = 0.6 / 2
            plt.rcParams[f"{var}.major.pad"] = 3.5 / 2
            plt.rcParams[f"{var}.minor.pad"] = 3.4 / 2
        plt.rcParams["axes.linewidth"] = 0.8 / 2
        plt.rcParams["grid.linewidth"] = 0.8 / 2
        plt.rcParams["lines.linewidth"] = 2 / 2
        plt.rcParams["lines.markersize"] = 6 / 2

    @staticmethod
    def fig_half():
        plt.rcParams["figure.figsize"] = figsize_half
        plt.rcParams["figure.dpi"] = fig_dpi
        plt.rcParams["font.size"] = font_small
        plt.rcParams["axes.titlesize"] = font_small
        plt.rcParams["axes.labelsize"] = font_medium
        plt.rcParams["xtick.labelsize"] = font_small
        plt.rcParams["ytick.labelsize"] = font_small
        plt.rcParams["legend.fontsize"] = font_small
        plt.rcParams["figure.titlesize"] = font_large

        for var in ["xtick", "ytick"]:
            plt.rcParams[f"{var}.major.size"] = 3.5
            plt.rcParams[f"{var}.minor.size"] = 2
            plt.rcParams[f"{var}.major.width"] = 0.8
            plt.rcParams[f"{var}.minor.width"] = 0.6
            plt.rcParams[f"{var}.major.pad"] = 3.5
            plt.rcParams[f"{var}.minor.pad"] = 3.4
        plt.rcParams["axes.linewidth"] = 0.8
        plt.rcParams["grid.linewidth"] = 0.8
        plt.rcParams["lines.linewidth"] = 2
        plt.rcParams["lines.markersize"] = 6

    @staticmethod
    def fig_full():
        plt.rcParams["figure.figsize"] = figsize_full
        plt.rcParams["figure.dpi"] = fig_dpi
        plt.rcParams["font.size"] = font_medium
        plt.rcParams["axes.titlesize"] = font_medium
        plt.rcParams["axes.labelsize"] = font_large
        plt.rcParams["xtick.labelsize"] = font_medium
        plt.rcParams["ytick.labelsize"] = font_medium
        plt.rcParams["legend.fontsize"] = font_medium
        plt.rcParams["figure.titlesize"] = font_huge

    # TODO(WAN): yet to make a full figure.


class GymConfigPlot:
    @staticmethod
    def load_model_eval(_expt_name: str) -> pd.DataFrame:
        return pd.read_parquet(Config.SAVE_PATH_EVAL / _expt_name)

    @staticmethod
    def read_runtime(_expt_name: str) -> float:
        df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / _expt_name / "0.parquet")
        return df.groupby(["Query Num"]).first()["Execution Time (ms)"].sum()

    @staticmethod
    def read_training_time(_expt_name: str) -> float:
        return pd.read_pickle(Config.SAVE_PATH_MODEL / _expt_name / "training_time.pkl")["Training Time (s)"]

    @staticmethod
    def generate_plot(plot_suffix, expt_names, plot_names=None):
        if plot_names is None:
            plot_names = expt_names
        assert len(plot_names) == len(expt_names)

        mae_s = []
        runtime_s = []
        training_time_s = []
        index = []
        for expt_name, plot_name in zip(expt_names, plot_names):
            metrics = GymConfigPlot.load_model_eval(expt_name)
            mae_s.append(metrics["Diff (ms)"].mean() / 1e3)
            runtime_s.append(GymConfigPlot.read_runtime(expt_name))
            training_time_s.append(GymConfigPlot.read_training_time(expt_name))
            index.append(plot_name)

        Config.SAVE_PATH_PLOT.mkdir(parents=True, exist_ok=True)
        PlotConfig.fig_half()
        fig, ax = plt.subplots(1, 1)
        df = pd.DataFrame({"MAE (s)": mae_s}, index=index)
        ax = df.plot.bar(ax=ax, rot=0, legend=False)
        ax.set_ylabel("Mean Absolute Error (s)")
        plt.tight_layout()
        fig.savefig(Config.SAVE_PATH_PLOT / f"accuracy_{plot_suffix}.pdf")
        plt.close(fig)

        PlotConfig.fig_half()
        fig, ax = plt.subplots(1, 1)
        df = pd.DataFrame(
            {
                "Runtime (s)": runtime_s,
                # "Training Time (s)": training_time_s,
            },
            index=index,
        )
        ax = df.plot.bar(stacked=True, ax=ax, rot=0, legend=False)
        ax.set_ylabel("Time (s)")
        plt.tight_layout()
        fig.savefig(Config.SAVE_PATH_PLOT / f"runtime_{plot_suffix}.pdf")
        plt.close(fig)

    @staticmethod
    def generate_plot_runtime_by_operator(plot_suffix, expt_names, plot_names=None):
        if plot_names is None:
            plot_names = expt_names
        assert len(plot_names) == len(expt_names)

        ndtt = "Nyoom Differenced Total Time (ms)"
        ntp = "Nyoom Tuples Processed"

        joined_df = None
        for expt_name, plot_name in zip(expt_names, plot_names):
            df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / expt_name / "0.parquet")
            df = (
                df.groupby("Node Type")[[ndtt, ntp]]
                .sum()
                .rename(columns={ndtt: f"{plot_name} {ndtt}", ntp: f"{plot_name} {ntp}"})
            )
            joined_df = df if joined_df is None else joined_df.join(df, how="outer")

        for node_type, row in joined_df.iterrows():
            x_pos = range(len(plot_names))
            y_val = []
            annotate_val = []
            for plot_name in plot_names:
                ndtt_val = row[f"{plot_name} {ndtt}"]
                ntp_val = row[f"{plot_name} {ntp}"]
                y_val.append(ndtt_val)
                if pd.isna(ntp_val):
                    annotate_val.append("")
                else:
                    annotate_val.append(str(int(ntp_val)))

            fig, ax = plt.subplots(1, 1)
            bar_container = ax.bar(
                x_pos,
                y_val,
                width=0.8,
                label=plot_names,
            )
            ax.bar_label(bar_container, labels=annotate_val)

            plt.xticks(x_pos, plot_names)
            plt.ylabel("Time (ms)")

            plt.tight_layout()
            plt.savefig(Config.SAVE_PATH_PLOT / f"runtime_by_operator_{plot_suffix}_{node_type}.pdf")
            plt.close(fig)
