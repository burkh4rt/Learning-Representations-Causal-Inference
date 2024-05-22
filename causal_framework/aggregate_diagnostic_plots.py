"""
When the true conditional average treatment effect is unavailable,
as would generally be the case for any real-life dataset including
data from controlled randomized a/b tests, we can still
get some sense of model effectiveness by generating
aggregated plots of average predicted and realized treatment
effect vs. predicted treatment effect quantile.
The Plotter object does this.
"""

from __future__ import annotations

import re
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc


class Plotter(object):
    def __init__(
        self,
        file_name: str,
        *,
        outcome_col: str = None,
        treatment_col: str = None,
        feature_col: list[str] = None,
        prediction_col: list[str] = None,
    ):
        self.file_path = file_name
        self.df = pd.read_csv(self.file_path)
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.feature_col = feature_col
        self.prediction_col = prediction_col

    def set_control(self, control) -> Plotter:
        """
        Replace the treatment column with a integer column `isChallenger`
        describing if each instance received the challenger treatment (1)
        or not (0)
        """
        self.df["isChallenger"] = (
            self.df[self.treatment_col]
            .apply(lambda x: (x != control))
            .astype("int")
        )
        self.df = self.df.drop(columns=self.treatment_col)
        self.treatment_col = "isChallenger"
        return self

    def generate_plot_by_binned_predictions(
        self,
        bins: int,
        predictor: str,
        *,
        title: str = None,
        xlabel: str = "bins",
        ylabel: str = "value",
        output_file_name: str = None,
        suffix: str = "",
        predictor_label: str = None,
        verbosity: int = 0,
        xlim: tuple[Number, Number] = None,
        ylim: tuple[Number, Number] = None,
    ):
        """
        Plot predictions and realizations aggregated by prediction quantile
        :param bins: number of bins to use
        :param predictor: name of column containing predictor
        :param title: title for plot, overrides default
        :param xlabel: label for x-axis
        :param ylabel: label for y-axis
        :param output_file_name: file name for output plot, overrides default
        :param suffix: adds to the end of the file name if output_file_name is
            not specified
        :param predictor_label: used in place of predictor name string in plot
            legend if specified
        :param verbosity: provides further information if set to >0
        :param xlim: specify x-axis limits for plot if desired
        :param ylim: specify y-axis limits for plot if desired
        :return: saves a plot as described
        """

        if predictor not in self.df.columns:
            print("Predictor should be a column in the data.")
        binned_variate = pd.qcut(self.df[predictor], bins)
        x_range = np.arange(1, 1 + bins)

        binned_preds = self.df.groupby(binned_variate, dropna=False).agg(
            {predictor: "mean"}
        )

        binned_outcome_1 = (
            self.df[self.df.isChallenger == 1]
            .groupby(binned_variate[self.df.isChallenger == 1], dropna=False)
            .agg({self.outcome_col: "mean"})
        )
        binned_outcome_0 = (
            self.df[self.df.isChallenger == 0]
            .groupby(binned_variate[self.df.isChallenger == 0], dropna=False)
            .agg({self.outcome_col: "mean"})
        )
        binned_outcomes = binned_outcome_1 - binned_outcome_0

        font = {"family": "serif", "serif": ["cmr10"]}
        rc("font", **font)
        rc("text", usetex=True)

        fig, ax = plt.subplots()

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.plot(x_range, binned_outcomes, label="realized")

        plt.plot(
            x_range,
            binned_preds,
            linestyle="dashed",
            label=predictor if predictor_label is None else predictor_label,
        )

        if verbosity > 0:
            mse = np.sqrt(
                np.mean(
                    np.square(
                        binned_preds.to_numpy().ravel()
                        - binned_outcomes.to_numpy().ravel()
                    )
                )
            )
            print(f"{predictor.title()} root mean sq. err. = {mse:.2f}")
            print(
                f"{predictor.title()} final bin real = "
                f"{binned_outcomes.to_numpy()[-1].round(2)[0]}"
            )

        handles, labels = ax.get_legend_handles_labels()
        unique_labels_dict = dict(zip(labels, handles))
        ax.legend(
            unique_labels_dict.values(),
            unique_labels_dict.keys(),
            fontsize="large",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(x_range)
        ax.set_title(
            f"Predictions and Realizations vs Binned {predictor}"
            if title is None
            else title
        )

        stringify = lambda s: re.sub(
            "_+", "_", re.sub("[^0-9a-zA-Z_]", "_", s.lower())
        ).strip(" _")
        plt.savefig(
            "preds_and_realizations_vs_binned"
            + f"_{stringify(predictor)}_{bins}_nbins{suffix}.pdf"
            if output_file_name is None
            else output_file_name,
            format="pdf",
        )

        # plt.show()
        plt.close(fig)

    def __del__(self):
        """closes any plots left open"""
        plt.close("all")
