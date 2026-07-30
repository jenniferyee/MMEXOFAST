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
        fitter.datasets = [SimpleNamespace(time=np.array(times))]
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
