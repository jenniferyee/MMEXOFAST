import unittest

import matplotlib

matplotlib.use("Agg")
import MulensModel
import numpy as np
import matplotlib.pyplot as plt

from mmexofast import AnomalyClassifier
from mmexofast import BellTemplateFitter


class TestAnomalyClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = AnomalyClassifier()

    def test_dip(self):
        dip_params = {
            "t_0": 2457942.6,
            "t_E": 47.0,
            "u_0": 0.25,
            "dmag": 0.25,
            "dt": 1.0,
            "t_pl": 2457958.7,
        }  # KB171194
        # Minimal residuals (not used for dmag>0 branch)
        time = np.linspace(0.0, 10.0, 50)
        flux = np.zeros_like(time)
        err = np.full_like(time, 0.01)
        residuals = MulensModel.MulensData([time, flux, err], phot_fmt="flux")

        result = self.classifier.classify(residuals, dip_params)
        print(f"Anomaly classification result for dip: {result}")
        assert result == "dip"

    def test_bump(self):
        bump_params = {
            "t_0": 2453582.7281740606,
            "u_0": 0.355227507989543,
            "t_E": 11.106795114521415,
            "dmag": -0.1,
            "dt": 0.5,
            "t_pl": 2453592.85,
        }  # OB053901,
        # Create a single-bump residual (should prefer 1-bell -> 'bump')
        time = np.linspace(bump_params["t_pl"] - 10 * bump_params["dt"], bump_params["t_pl"] + 10 * bump_params["dt"], 200)
        width = 0.2
        amp = 0.2
        flux = amp * np.exp(-0.5 * ((time - bump_params["t_pl"]) / width) ** 2)
        err = np.full_like(time, 0.01)
        residuals = MulensModel.MulensData([time, flux, err], phot_fmt="flux")

        result = self.classifier.classify(residuals, bump_params)
        assert result == "bump"

    def test_hm(self):
        hm_params = {
            "t_0": 2453480.68,
            "t_E": 73.9,
            "u_0": 0.023,
            "dmag": -0.3,
            "dt": 1.5,
            "t_pl": 2453480.6,
        }  # OB05071
        # small-ish u0 leads to high_mag in the classifier logic for dmag<0
        time = np.linspace(hm_params["t_pl"] - 2.0 * hm_params["dt"], hm_params["t_pl"] + 2.0 * hm_params["dt"], 200)
        flux = -0.1 * np.exp(-0.5 * ((time - hm_params["t_pl"]) / 0.3) ** 2)
        err = np.full_like(time, 0.01)
        residuals = MulensModel.MulensData([time, flux, err], phot_fmt="flux")

        result = self.classifier.classify(residuals, hm_params)
        assert result == "high_mag"


class TestTwoBellTemplateFitter(unittest.TestCase):
    def test_run(self):
        time = np.linspace(0.0, 10.0, 200)
        t_pl = 5.0
        dt = 2.0
        t_1 = t_pl - 0.5 * dt
        t_2 = t_pl + 0.5 * dt
        width = 0.35
        amp_1 = 1.4
        amp_2 = 0.9
        offset = 0.15
        flux = (
            offset
            + amp_1 * np.exp(-0.5 * ((time - t_1) / width) ** 2)
            + amp_2 * np.exp(-0.5 * ((time - t_2) / width) ** 2)
        )
        err = np.full_like(time, 0.03)
        data = MulensModel.MulensData([time, flux, err], phot_fmt="flux")

        fitter = BellTemplateFitter([data], {"t_pl": t_pl, "dt": dt})
        best = fitter.run()

        np.testing.assert_allclose(best["t_1"], t_1)
        np.testing.assert_allclose(best["t_2"], t_2)
        np.testing.assert_allclose(best["amplitudes"][0], amp_1, rtol=0.15)
        np.testing.assert_allclose(best["amplitudes"][1], amp_2, rtol=0.15)
        np.testing.assert_allclose(best["width"], width, rtol=0.25)
        np.testing.assert_allclose(best["dt"], dt, rtol=0.2)

    def test_run_ignores_out_of_window_outlier(self):
        time = np.linspace(0.0, 10.0, 200)
        t_pl = 5.0
        dt = 2.0
        t_1 = t_pl - 0.5 * dt
        t_2 = t_pl + 0.5 * dt
        width = 0.35
        amp_1 = 1.4
        amp_2 = 0.9
        offset = 0.15
        flux = (
            offset
            + amp_1 * np.exp(-0.5 * ((time - t_1) / width) ** 2)
            + amp_2 * np.exp(-0.5 * ((time - t_2) / width) ** 2)
        )
        err = np.full_like(time, 0.03)

        time = np.append(time, 20.0)
        flux = np.append(flux, 50.0)
        err = np.append(err, 0.03)
        data = MulensModel.MulensData([time, flux, err], phot_fmt="flux")

        fitter = BellTemplateFitter([data], {"t_pl": t_pl, "dt": dt})
        best = fitter.run()

        np.testing.assert_allclose(best["t_1"], t_1)
        np.testing.assert_allclose(best["t_2"], t_2)
        np.testing.assert_allclose(best["amplitudes"][0], amp_1, rtol=0.15)
        np.testing.assert_allclose(best["amplitudes"][1], amp_2, rtol=0.15)
        np.testing.assert_allclose(best["width"], width, rtol=0.25)
        np.testing.assert_allclose(best["dt"], dt, rtol=0.2)

    def test_run_with_three_bells(self):
        time = np.linspace(0.0, 12.0, 240)
        t_pl = 6.0
        dt = 4.0
        centers = [t_pl - 0.5 * dt, t_pl, t_pl + 0.5 * dt]
        width = 0.2
        amps = [1.0, 0.6, 0.8]
        offset = 0.05
        flux = np.full_like(time, offset)
        for amp, center in zip(amps, centers):
            flux += amp * np.exp(-0.5 * ((time - center) / width) ** 2)
        err = np.full_like(time, 0.03)
        data = MulensModel.MulensData([time, flux, err], phot_fmt="flux")

        fitter = BellTemplateFitter([data], {"t_pl": t_pl, "dt": dt}, n_bells=3)
        best = fitter.run()

        assert len(best["amplitudes"]) == 3
        np.testing.assert_allclose(best["centers"], centers)
        np.testing.assert_allclose(best["amplitudes"][0], amps[0], rtol=0.2)
        np.testing.assert_allclose(best["amplitudes"][1], amps[1], rtol=0.2)
        np.testing.assert_allclose(best["amplitudes"][2], amps[2], rtol=0.2)
        np.testing.assert_allclose(best["width"], width, rtol=0.3)
        np.testing.assert_allclose(best["dt"], dt, rtol=0.2)

    def test_run_with_fitted_centers_and_bounds(self):
        time = np.linspace(0.0, 10.0, 240)
        t_pl = 5.0
        dt = 2.0
        centers = [4.25, 5.05, 5.75]
        width = 0.2
        amps = [1.2, 0.7, 1.0]
        offset = 0.05

        flux = np.full_like(time, offset)
        for amp, center in zip(amps, centers):
            flux += amp * np.exp(-0.5 * ((time - center) / width) ** 2)

        err = np.full_like(time, 0.03)
        data = MulensModel.MulensData([time, flux, err], phot_fmt="flux")

        fitter = BellTemplateFitter(
            [data],
            {"t_pl": t_pl, "dt": dt},
            n_bells=3,
            fit_centers=True,
        )
        best = fitter.run()

        edges = np.linspace(t_pl - 0.5 * dt, t_pl + 0.5 * dt, 4)
        for i, center in enumerate(best["centers"]):
            assert edges[i] <= center <= edges[i + 1]

        np.testing.assert_allclose(best["centers"], centers, atol=0.12)
        np.testing.assert_allclose(best["width"], width, rtol=0.35)

    def test_plot_fit(self):
        time = np.linspace(0.0, 10.0, 200)
        t_pl = 5.0
        dt = 2.0
        t_1 = t_pl - 0.5 * dt
        t_2 = t_pl + 0.5 * dt
        width = 0.35
        amp_1 = 1.4
        amp_2 = -0.9
        offset = 0.15
        flux = (
            offset
            + amp_1 * np.exp(-0.5 * ((time - t_1) / width) ** 2)
            + amp_2 * np.exp(-0.5 * ((time - t_2) / width) ** 2)
        )
        err = np.full_like(time, 0.03)
        data = MulensModel.MulensData([time, flux, err], phot_fmt="flux")

        fitter = BellTemplateFitter([data], {"t_pl": t_pl, "dt": dt})
        fig, ax_fit, ax_resid = fitter.plot_fit()

        assert fig is not None
        assert ax_fit is not None
        assert ax_resid is not None
        assert len(ax_fit.lines) >= 7
        assert len(ax_resid.lines) >= 2
        plt.close(fig)
