import matplotlib.pyplot as plt
import MulensModel
import numpy as np
from scipy.optimize import curve_fit

from mmexofast.fitters import BellTemplateFitter


class AnomalyClassifier(object):
    """
    Classify a microlensing anomaly into one of the supported labels.

    The classifier returns one of ``'bump'``, ``'dip'``,
    ``'caustic_crossing'``, or ``'high_mag'`` based on the anomaly light
    curve and parameters.
    """

    def __init__(self):
        pass
    def plot_bell_fits(self):
        """
        Plot the best-fit 1- and 2-bell templates for the anomaly classification.
        """
        # Use one shared fit panel and separate residual panels for each model.
        fig1, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
        fig1, ax_fit1, ax_resid1 = self._template_fit1.plot_fit(show=False,
            t_range=(
                self.lc_parameters["t_pl"] - 2 * self.lc_parameters["dt"],
                self.lc_parameters["t_pl"] + 2 * self.lc_parameters["dt"],
            ), fig=fig1, ax_fit=axes[0], ax_resid=axes[1])

        fig2, ax_fit2, ax_resid2 = self._template_fit2.plot_fit(show=False,
            t_range=(
                self.lc_parameters["t_pl"] - 2* self.lc_parameters["dt"],
                self.lc_parameters["t_pl"] + 2* self.lc_parameters["dt"],
            ), ax_fit=ax_fit1, ax_resid=axes[2], fig=fig1, model_color="C6",
            vline_color="C4", residual_color="C6")

        label = "bump" if self._template_fit1.best["chi2"] <= self._template_fit2.best["chi2"] else "caustic_crossing"
        plt_title = f"Anomaly classification: {label}\n"
        plt_title += f"1-bell chi2: {self._template_fit1.best['chi2']:.2f}, 2-bell chi2: {self._template_fit2.best['chi2']:.2f}"
        axes[0].set_title(plt_title)
        fig1.tight_layout()
        fig1.show()
        return fig1

    def classify(self, residuals, lc_parameters,):
        """
        Use the lightcurve and anomaly properties to determine what kind of fit is needed.

        Parameters
        ----------
        params : dict
            Results of AnomalyPropertyEstimator.get_anomaly_lc_parameters()

        Returns
        -------
        str
            One of 'bump', 'dip', 'caustic_crossing', 'high_mag'
        """

        self.lc_parameters = lc_parameters
        self.residuals = residuals

        if lc_parameters["dmag"] < 0:
            self._template_fit1 = BellTemplateFitter(residuals, lc_parameters, n_bells=1)
            self._template_fit1.run()
            print(f"1-bell template fit results: {self._template_fit1.best['chi2']}")
            # print(f"best: {self._template_fit1.best}")

            self._template_fit2 = BellTemplateFitter(self.residuals, self.lc_parameters, n_bells=2)
            self._template_fit2.run()
            print(f"2-bell template fit results: {self._template_fit2.best['chi2']}")
                    # print(f"best: {self._template_fit2.best}")


        if np.abs(lc_parameters["u_0"]) < 0.01:
            return "high_mag"

        if lc_parameters["dmag"] < 0:
            if np.abs(lc_parameters["u_0"]) > 0.05:
                if self._template_fit1.best["chi2"] < self._template_fit2.best["chi2"]:
                    return "bump"
                else:
                    return "caustic_crossing"
            else:
                return "high_mag"

        if lc_parameters["dmag"] > 0:
            return "dip"
