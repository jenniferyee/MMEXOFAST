import json
import unittest
from types import SimpleNamespace

import MulensModel
import numpy as np

from mmexofast import MMEXOFASTFitter


class TestMMEXOFASTFitter(unittest.TestCase):
    def test_fit(self):
        self.skipTest("Not Implemented")

    def test_do_ef_grid_search(self):
        self.skipTest("Not Implemented")

    def test_get_initial_pspl_params(self):
        self.skipTest("Not Implemented")

    def test_do_sfit(self):
        self.skipTest("Not Implemented")

    def test_do_mmexofast_fit(self):
        self.skipTest("Not Implemented")

    def test_set_datasets_with_anomaly_masked(self):
        self.skipTest("Not Implemented")

    def test_get_residuals_mask(self):
        self.skipTest("Not Implemented")

    def test_refine_pspl_params(self):
        self.skipTest("Not Implemented")

    def test_set_residuals(self):
        self.skipTest("Not Implemented")

    def test_do_af_grid_search(self):
        self.skipTest("Not Implemented")

    def test_get_dmag(self):
        self.skipTest("Not Implemented")

    def test_get_initial_2L1S_params(self):
        self.skipTest("Not Implemented")

    def test_residuals(self):
        self.skipTest("Not Implemented")

    def test_residuals_setter(self):
        self.skipTest("Not Implemented")

    def test_masked_datasets(self):
        self.skipTest("Not Implemented")

    def test_masked_datasets_setter(self):
        self.skipTest("Not Implemented")

    def test_best_ef_grid_point(self):
        self.skipTest("Not Implemented")

    def test_best_ef_grid_point_setter(self):
        self.skipTest("Not Implemented")

    def test_pspl_params(self):
        self.skipTest("Not Implemented")

    def test_pspl_params_setter(self):
        self.skipTest("Not Implemented")

    def test_best_af_grid_point(self):
        self.skipTest("Not Implemented")

    def test_best_af_grid_point_setter(self):
        self.skipTest("Not Implemented")

    def test_binary_params(self):
        self.skipTest("Not Implemented")

    def test_binary_params_setter(self):
        self.skipTest("Not Implemented")

    def test_results(self):
        self.skipTest("Not Implemented")

    def test_results_setter(self):
        self.skipTest("Not Implemented")


class TestInitializeExozippy(unittest.TestCase):
    """
    The EXOZIPPy handoff: coordinates, and epochs on a full JD scale.

    Built with __new__ and the handful of attributes the method reads,
    rather than running a fit, so these stay fast and independent of the
    fitting machinery.
    """

    def _make_fitter(self, fit_type, records=(), times=(2455000.0,)):
        fitter = MMEXOFASTFitter.__new__(MMEXOFASTFitter)
        fitter.fit_type = fit_type
        fitter.renorm_factors = {"OGLE": 1.0}
        fitter.mag_methods = []
        fitter.coords = None
        fitter.datasets = [
            SimpleNamespace(
                time=np.array(times),
                bad=np.zeros(len(times), dtype=bool),
                plot_properties={"label": "OGLE_I.dat"},
            )
        ]
        fitter.all_fit_results = {}
        keys = []
        for i, params in enumerate(records):
            key = "key{0}".format(i)
            keys.append(key)
            fitter.all_fit_results[key] = SimpleNamespace(
                params=params, sigmas=None
            )
        fitter._iter_parallax_point_lens_keys = lambda: keys
        return fitter

    def test_full_jd_data_is_left_alone(self):
        """Data already in full JD needs no shift."""
        fitter = self._make_fitter(
            "point_lens",
            records=[{"t_0": 2455000.0, "u_0": 0.1, "t_E": 20.0}],
            times=(2455010.0,),
        )
        result = fitter.initialize_exozippy()

        self.assertEqual(result["jd_offset"], 0.0)
        self.assertEqual(result["fits"][0]["parameters"]["t_0"], 2455000.0)

    def test_reduced_hjd_is_converted(self):
        """HJD' = HJD - 2450000 is detected from the data and undone."""
        fitter = self._make_fitter(
            "point_lens",
            records=[{"t_0": 5000.0, "u_0": 0.1, "t_E": 20.0}],
            times=(5010.0,),
        )
        result = fitter.initialize_exozippy()

        self.assertEqual(result["jd_offset"], 2450000.0)
        self.assertAlmostEqual(
            result["fits"][0]["parameters"]["t_0"], 2455000.0
        )

    def test_shifts_every_epoch_not_just_t_0(self):
        """
        t_0_par and friends are epochs too. Shifting t_0 alone would leave
        a parallax fit internally inconsistent.
        """
        fitter = self._make_fitter(
            "point_lens",
            records=[
                {
                    "t_0": 5000.0,
                    "t_0_par": 5001.0,
                    "u_0": 0.1,
                    "t_E": 20.0,
                    "pi_E_N": 0.1,
                }
            ],
            times=(5010.0,),
        )
        params = fitter.initialize_exozippy()["fits"][0]["parameters"]

        self.assertAlmostEqual(params["t_0"], 2455000.0)
        self.assertAlmostEqual(params["t_0_par"], 2455001.0)

    def test_durations_are_not_shifted(self):
        """t_E and t_star are invariant under a change of time origin."""
        fitter = self._make_fitter(
            "point_lens",
            records=[{"t_0": 5000.0, "u_0": 0.1, "t_E": 20.0, "t_star": 0.3}],
            times=(5010.0,),
        )
        params = fitter.initialize_exozippy()["fits"][0]["parameters"]

        self.assertEqual(params["t_E"], 20.0)
        self.assertEqual(params["t_star"], 0.3)

    def test_does_not_mutate_the_stored_record(self):
        """The fit record keeps the epochs it was fitted with."""
        record = {"t_0": 5000.0, "u_0": 0.1, "t_E": 20.0}
        fitter = self._make_fitter(
            "point_lens", records=[record], times=(5010.0,)
        )
        fitter.initialize_exozippy()

        self.assertEqual(record["t_0"], 5000.0)

    def test_coords_reported_as_string(self):
        fitter = self._make_fitter("point_lens")
        fitter.coords = MulensModel.Coordinates("17:54:19.2 -30:22:38")
        result = fitter.initialize_exozippy()

        self.assertEqual(result["coords"], "17:54:19.20 -30:22:38.00")

    def test_coords_none_when_not_supplied(self):
        fitter = self._make_fitter("point_lens")
        self.assertIsNone(fitter.initialize_exozippy()["coords"])

    def test_no_datasets_means_no_shift(self):
        """Without data there is nothing to infer a convention from."""
        fitter = self._make_fitter("point_lens")
        fitter.datasets = []
        self.assertEqual(fitter.initialize_exozippy()["jd_offset"], 0.0)

    def test_output_is_json_serializable(self):
        """
        The result is written straight to JSON with no custom encoder, so
        everything in it has to survive json.dumps -- including the numpy
        floats a real fit produces.
        """
        fitter = self._make_fitter(
            "point_lens",
            records=[
                {
                    "t_0": np.float64(5000.0),
                    "u_0": np.float64(0.1),
                    "t_E": np.float64(20.0),
                }
            ],
            times=(5010.0,),
        )
        fitter.coords = MulensModel.Coordinates("17:54:19.2 -30:22:38")

        loaded = json.loads(json.dumps(fitter.initialize_exozippy()))
        self.assertAlmostEqual(
            loaded["fits"][0]["parameters"]["t_0"], 2455000.0
        )
        self.assertEqual(loaded["jd_offset"], 2450000.0)

    def test_unsupported_fit_type_raises(self):
        fitter = self._make_fitter("something_else")
        with self.assertRaises(NotImplementedError):
            fitter.initialize_exozippy()


class TestExozippyExcludedPoints(unittest.TestCase):
    """
    The outlier mask handed to EXOZIPPy, reported as indices rather than an
    n_data-length boolean array.
    """

    def _make_fitter(self, bad, times=None, label="OGLE_I.dat"):
        n = len(bad)
        if times is None:
            times = 2455000.0 + np.arange(n, dtype=float)
        fitter = MMEXOFASTFitter.__new__(MMEXOFASTFitter)
        fitter.datasets = [
            SimpleNamespace(
                bad=np.array(bad, dtype=bool),
                time=np.asarray(times, dtype=float),
                plot_properties={"label": label},
            )
        ]
        return fitter

    def test_reports_indices_of_excluded_points(self):
        fitter = self._make_fitter([False, True, False, False, True])
        record = fitter._exozippy_excluded_points(0.0)["OGLE_I.dat"]

        self.assertEqual(record["indices"], [1, 4])
        self.assertEqual(record["n_data"], 5)

    def test_reports_times_of_excluded_points(self):
        fitter = self._make_fitter([False, True, False])
        record = fitter._exozippy_excluded_points(0.0)["OGLE_I.dat"]

        self.assertEqual(record["times"], [2455001.0])

    def test_times_share_the_epoch_system_of_the_fits(self):
        """
        Reported times carry the same jd_offset as the fitted epochs, so the
        two cannot end up in different time systems.
        """
        fitter = self._make_fitter([False, True], times=[5000.0, 5001.0])
        record = fitter._exozippy_excluded_points(2450000.0)["OGLE_I.dat"]

        self.assertEqual(record["times"], [2455001.0])

    def test_empty_when_nothing_excluded(self):
        fitter = self._make_fitter([False, False, False])
        record = fitter._exozippy_excluded_points(0.0)["OGLE_I.dat"]

        self.assertEqual(record["indices"], [])
        self.assertEqual(record["times"], [])
        self.assertEqual(record["n_data"], 3)

    def test_keyed_per_dataset_like_errfacs(self):
        fitter = self._make_fitter([False, True])
        fitter.datasets.append(
            SimpleNamespace(
                bad=np.array([True, False]),
                time=np.array([2455010.0, 2455011.0]),
                plot_properties={"label": "MOA_r.dat"},
            )
        )
        excluded = fitter._exozippy_excluded_points(0.0)

        self.assertEqual(sorted(excluded), ["MOA_r.dat", "OGLE_I.dat"])
        self.assertEqual(excluded["MOA_r.dat"]["indices"], [0])

    def test_indices_are_plain_ints_for_json(self):
        """numpy ints do not serialize; these must be Python ints."""
        fitter = self._make_fitter([False, True])
        record = fitter._exozippy_excluded_points(0.0)["OGLE_I.dat"]

        self.assertIsInstance(record["indices"][0], int)
        self.assertNotIsInstance(record["indices"][0], np.integer)
        json.dumps(record)


class TestSatelliteData(unittest.TestCase):
    def setUp(self):
        self.ground_data = None
        self.spitzer_data = None
        self.kepler_data = None
        self.skipTest(
            "Need to setup test datasets for this test (among other things)."
        )

    def do_test_file_list(self, files):
        self.skipTest("Not Implemented")
        fitter = MMEXOFASTFitter(files=files)
        assert fitter.n_loc == len(files)

    def _get_datasets(self, file_list):
        # Do something to read in the datasets
        raise NotImplementedError(
            "test datasets are not set up yet; see setUp()"
        )

    def do_test_datasets(self, file_list):
        self.skipTest("Not Implemented")
        datasets = self._get_datasets(file_list)
        fitter = MMEXOFASTFitter(datasets=datasets)
        assert fitter.n_loc == len(datasets)

    def test_gr_only(self):
        self.do_test_file_list(self.ground_data)
        self.do_test_datasets(self.ground_data)

    def test_spz_only(self):
        self.do_test_file_list(self.spitzer_data)
        self.do_test_datasets(self.spitzer_data)

    def test_gr_spz(self):
        files = [self.ground_data, self.spitzer_data]
        self.do_test_file_list(files)
        self.do_test_datasets(files)

    def test_kep_spz(self):
        files = [self.kepler_data, self.spitzer_data]
        self.do_test_file_list(files)
        self.do_test_datasets(files)

    def test_all(self):
        files = [self.ground_data, self.kepler_data, self.spitzer_data]
        self.do_test_file_list(files)
        self.do_test_datasets(files)


class TestBuildProtectedMask(unittest.TestCase):
    """
    Points protected from renormalization's outlier rejection: the anomaly
    window and everywhere the reference model magnifies above
    PEAK_PROTECTION_MAG.
    """

    T_0, U_0, T_E = 2455500.0, 0.1, 20.0

    def _make_fitter_and_event(self, anomaly_lc_params=None):
        model = MulensModel.Model(
            {"t_0": self.T_0, "u_0": self.U_0, "t_E": self.T_E}
        )
        times = self.T_0 + np.arange(-100.0, 100.0, 0.5)
        flux = 10.0 * model.get_magnification(times) + 2.0
        data = MulensModel.MulensData(
            [times, flux, 0.01 * np.ones_like(flux)],
            phot_fmt="flux",
            plot_properties={"label": "test.dat"},
        )
        event = MulensModel.Event(datasets=[data], model=model)
        event.fit_fluxes()
        fitter = MMEXOFASTFitter.__new__(MMEXOFASTFitter)
        fitter.intermediate_results = SimpleNamespace(
            anomaly_lc_params=anomaly_lc_params
        )
        return fitter, event, data

    def test_peak_is_protected_even_without_anomaly_params(self):
        fitter, event, data = self._make_fitter_and_event(None)

        mask = fitter._build_protected_mask(data, event, 0)

        mag = event.model.get_magnification(data.time)
        np.testing.assert_array_equal(
            mask, mag > MMEXOFASTFitter.PEAK_PROTECTION_MAG
        )
        self.assertTrue(mask.any())
        self.assertFalse(mask.all())

    def test_anomaly_window_is_protected(self):
        t_pl = self.T_0 + 60.0  # on the wing, model magnification < 1.1
        params = {"t_0": self.T_0, "t_pl": t_pl, "dt": 2.0}
        fitter, event, data = self._make_fitter_and_event(params)

        mask = fitter._build_protected_mask(data, event, 0)

        window = (data.time >= t_pl - 2.0) & (data.time <= t_pl + 2.0)
        self.assertTrue(window.any())
        self.assertTrue(np.all(mask[window]))

    def test_mask_is_union_of_anomaly_window_and_peak(self):
        t_pl = self.T_0 + 60.0
        params = {"t_0": self.T_0, "t_pl": t_pl, "dt": 2.0}
        fitter, event, data = self._make_fitter_and_event(params)

        mask = fitter._build_protected_mask(data, event, 0)

        mag = event.model.get_magnification(data.time)
        window = (data.time >= t_pl - 2.0) & (data.time <= t_pl + 2.0)
        expected = window | (mag > MMEXOFASTFitter.PEAK_PROTECTION_MAG)
        np.testing.assert_array_equal(mask, expected)

    def test_window_centered_on_t_pl_not_t_0(self):
        t_pl = self.T_0 + 60.0
        params = {"t_0": self.T_0, "t_pl": t_pl, "dt": 2.0}
        fitter, event, data = self._make_fitter_and_event(params)

        mask = fitter._build_protected_mask(data, event, 0)

        # Points near t_0 + dt but outside the peak-protection region would
        # only be masked if the window were (wrongly) centered on t_0; with
        # u_0=0.1, t_E=20 the A > 1.1 region ends near |t - t_0| ~ 31 d.
        wing = np.abs(data.time - (self.T_0 + 40.0)) < 1.0
        self.assertTrue(wing.any())
        self.assertFalse(np.any(mask[wing]))
