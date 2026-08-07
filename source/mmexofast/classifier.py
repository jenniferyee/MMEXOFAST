import numpy as np
import MulensModel
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class BellTemplateFitter:
    """
    Fit residuals with a simple multi-bell template.

    The bell centres are tied to the anomaly timing estimate and spread across
    the requested time span. The fit solves for one amplitude per bell, a
    shared width, and a constant offset.
    """

    def __init__(self, residuals, lc_parameters, n_bells=2, fit_centers=False):
        if isinstance(residuals, MulensModel.MulensData):
            residuals = [residuals]

        self.residuals = residuals
        self.lc_parameters = lc_parameters
        self.n_bells = max(int(n_bells), 1)
        self.fit_centers = bool(fit_centers)
        self.t_pl = float(lc_parameters.get("t_pl", None))
        self.best = None

    def _bell_centers(self, t_pl, dt):
        if self.n_bells == 1:
            return np.array([t_pl], dtype=float)
        offsets = np.linspace(-0.5 * dt, 0.5 * dt, self.n_bells)
        return t_pl + offsets

    def _center_bounds(self, t_pl, dt):
        # Keep each center in its own equal-width share of the dt span.
        edges = np.linspace(t_pl - 0.5 * dt, t_pl + 0.5 * dt, self.n_bells + 1)
        return edges[:-1], edges[1:]

    @staticmethod
    def _template(time, centers=None, width=None, offset=None, amplitudes=None, dt=None, t_pl=None, **kwargs):
        width = max(float(width), 1.0e-6)
        model = np.full_like(time, offset, dtype=float)
        for amplitude, center in zip(amplitudes, centers):
            model += amplitude * np.exp(-0.5 * ((time - center) / width) ** 2)
        model[time < t_pl - 0.5 * dt] = 0
        model[time > t_pl + 0.5 * dt] = 0
        return model

    def _stack_pspl_residuals(self):
        times = []
        fluxes = []
        errors = []

        for dataset in self.residuals:
            good = getattr(dataset, "good", None)
            if good is None:
                good = np.ones(len(dataset.time), dtype=bool)

            times.append(np.asarray(dataset.time)[good])
            fluxes.append(np.asarray(dataset.flux)[good])
            errors.append(np.asarray(dataset.err_flux)[good])

        time = np.hstack(times)
        flux = np.hstack(fluxes)
        err = np.hstack(errors)
        order = np.argsort(time)
        return time[order], flux[order], err[order]

    @staticmethod
    def _fit_window(time, flux, err, t_pl, dt):
        half_width = 5 * max(float(dt), 1.0e-6)
        window = (time >= t_pl - half_width) & (time <= t_pl + half_width)
        if not np.any(window):
            return time, flux, err
        return time[window], flux[window], err[window]

    def _set_time_parameters(self, time):
        """Set initial guess for the fit parameters based on the residuals and lightcurve parameters."""
        t_pl = float(self.lc_parameters.get("t_pl", np.median(time)))
        dt_0 = float(abs(self.lc_parameters.get("dt", np.ptp(time) / 6.0)))
        dt_0 = max(dt_0, 1.0e-6)
        return t_pl, dt_0

    def _set_shape_parameters(self, t_pl, dt_0):
        """Set initial guess for the fit parameters based on the residuals and lightcurve parameters."""
        centers = self._bell_centers(t_pl, dt_0)
        baseline = 0.0
        amplitudes = [float(self._flux[np.argmin(np.abs(self._time - center))] - baseline) for center in centers]
        width = max(dt_0 / 4.0, np.median(np.diff(self._time)) if len(self._time) > 1 else dt_0)
        return centers, amplitudes, width, baseline

    def _set_initial_guess(self, amplitudes, centers, width, baseline, dt_0):
        """Set initial guess for the fit parameters based on the residuals and lightcurve parameters."""
        p0 = amplitudes
        if self.fit_centers:
            p0 += [float(value) for value in centers]
        p0 += [width, baseline, dt_0]
        return p0

    def _set_bounds(self, t_pl, dt_0):
        """Set bounds for the fit parameters"""

        lower_bounds = [0.0] * self.n_bells
        upper_bounds = [np.inf] * self.n_bells

        if self.fit_centers:
            center_low, center_high = self._center_bounds(t_pl, dt_0)
            lower_bounds += [float(value) for value in center_low]
            upper_bounds += [float(value) for value in center_high]

        lower_bounds += [1.0e-6, -np.inf, 0.5 * dt_0]
        upper_bounds += [dt_0 / self.n_bells/self.n_bells, np.inf, 5.0 * dt_0]
        bounds = (lower_bounds, upper_bounds)
        return bounds

    def _set_results(self, popt, pcov, centers, model_flux):
        """Store the best-fit parameters and chi-squared value."""
        chi2 = float(np.sum(((self._flux - model_flux) / self._err) ** 2))

        index = self.n_bells
        fit_centers = centers
        if self.fit_centers:
            fit_centers = np.asarray(popt[index: index + self.n_bells], dtype=float)
            index += self.n_bells

        self.best = {
            "amplitudes": [float(value) for value in popt[: self.n_bells]],
            "width": float(popt[index]),
            "offset": float(popt[index + 1]),
            "centers": [float(center) for center in fit_centers],
            "dt": float(popt[index + 2]),
            "chi2": chi2,
            "covariance": pcov,
        }
        for bell_index, center in enumerate(self.best["centers"], start=1):
            self.best[f"t_{bell_index}"] = float(center)
            self.best[f"amp_{bell_index}"] = float(popt[bell_index - 1])

    def run(self):
        """Run the fit and store the best-fit parameters."""
        time, flux, err = self._stack_pspl_residuals()
        t_pl, dt_0 = self._set_time_parameters(time)
        self._time, self._flux, self._err = self._fit_window(time, flux, err, t_pl, dt_0)

        centers, amplitudes, width, baseline = self._set_shape_parameters(t_pl, dt_0)
        p0 = self._set_initial_guess(amplitudes, centers, width, baseline, dt_0)
        bounds = self._set_bounds(t_pl, dt_0)

        # print(f"Initial guess: {p0}")
        # print(f"Bounds: {bounds}")

        def model(x, *params):
            fit_amplitudes = params[: self.n_bells]
            index = self.n_bells
            fit_centers = centers
            if self.fit_centers:
                fit_centers = params[index: index + self.n_bells]
                index += self.n_bells
            fit_width = params[index]
            fit_offset = params[index + 1]
            fit_dt = params[index + 2]
            return self._template(x, fit_centers, fit_width, fit_offset, fit_amplitudes, fit_dt, self.t_pl)

        popt, pcov = curve_fit(model, self._time, self._flux, p0=p0, sigma=self._err, absolute_sigma=True, bounds=bounds, maxfev=50000)

        model_flux = model(self._time, *popt)
        self._set_results(popt, pcov, centers, model_flux)
        return self.best

    def plot_fit(self, best=None, fig=None, ax_fit=None, ax_resid=None, show=False, t_range=None, data_color="k",
        model_color="C1", vline_color="C2", residual_color="C1",):
        if self.best is None:
            self.run()

        time, flux, err = self._stack_pspl_residuals()
        # Build model_flux by passing the expected keyword arguments to _template
        model_flux = self._template(
            time,
            centers=self.best["centers"],
            width=self.best["width"],
            offset=self.best["offset"],
            amplitudes=self.best["amplitudes"],
            dt=self.best["dt"],
            t_pl=self.t_pl,
        )
        fit_residuals = flux - model_flux

        if ax_fit is None or ax_resid is None:
            fig, (ax_fit, ax_resid) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, 
                                                   gridspec_kw={"height_ratios": [3, 1]},)
        elif fig is None:
            fig = ax_fit.figure

        ax_fit.errorbar(time, flux, yerr=err, fmt=".", color=data_color, alpha=0.6, label="data")
        ax_fit.plot(time, model_flux, color=model_color, lw=2, label=f"{self.n_bells}-bell fit")
        for center in self.best["centers"]:
            ax_fit.axvline(center, color=vline_color, ls="--", alpha=0.7)
            ax_fit.axvline(center - self.best["width"], color=vline_color, ls=":", alpha=0.5)
            ax_fit.axvline(center + self.best["width"], color=vline_color, ls=":", alpha=0.5)
        ax_fit.set_ylabel("Residual flux")
        if t_range is not None:
            ax_fit.set_xlim(t_range)
        ax_fit.legend()

        ax_resid.errorbar(time, fit_residuals, yerr=err, fmt=".", color=residual_color, alpha=0.6,
            label=f"{self.n_bells}-bell residuals")
        ax_resid.axhline(0.0, color="0.3", ls="--", lw=1)
        ax_resid.set_xlabel("Time")
        ax_resid.set_ylabel("Data - fit")
        if t_range is not None:
            ax_resid.set_xlim(t_range)
        ax_resid.legend()

        if show:
            plt.show()

        return fig, ax_fit, ax_resid


class AnomalyClassifier(object):
    """
    Classifies a microlensing event anomaly as 'close', 'wide', or 'high_mag'.

    Uses lightcurve and anomaly parameters from
    AnomalyPropertyEstimator.get_anomaly_lc_parameters() to determine
    the classification of the event.
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
            One of 'bump', 'caustic_crossing', 'high_mag'
        """

        self.lc_parameters = lc_parameters
        self.residuals = residuals

        if lc_parameters["dmag"] < 0:
            self._template_fit1 = BellTemplateFitter(residuals, lc_parameters, n_bells=1)
            self._template_fit1.run()
            print(f"{1}-bell template fit results: {self._template_fit1.best['chi2']}")
            # print(f"best: {self._template_fit1.best}")

            self._template_fit2 = BellTemplateFitter(self.residuals, self.lc_parameters, n_bells=2)
            self._template_fit2.run()
            print(f"{2}-bell template fit results: {self._template_fit2.best['chi2']}")
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
            return "close"
