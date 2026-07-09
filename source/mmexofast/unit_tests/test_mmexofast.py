import unittest
from types import SimpleNamespace
from mmexofast.mmexofast import MMEXOFASTFitter

class TestMMEXOFASTFitter(unittest.TestCase):

    def test_fit(self):
        self.skipTest('Not Implemented')

    def test_do_ef_grid_search(self):
        self.skipTest('Not Implemented')

    def test_get_initial_pspl_params(self):
        self.skipTest('Not Implemented')

    def test_do_sfit(self):
        self.skipTest('Not Implemented')

    def test_do_mmexofast_fit(self):
        self.skipTest('Not Implemented')

    def test_set_datasets_with_anomaly_masked(self):
        self.skipTest('Not Implemented')

    def test_get_residuals_mask(self):
        self.skipTest('Not Implemented')

    def test_refine_pspl_params(self):
        self.skipTest('Not Implemented')

    def test_set_residuals(self):
        self.skipTest('Not Implemented')

    def test_do_af_grid_search(self):

        self.skipTest('Not Implemented')

    def test_get_dmag(self):
        self.skipTest('Not Implemented')

    def test_get_initial_2L1S_params(self):
        self.skipTest('Not Implemented')

    def test_residuals(self):
        self.skipTest('Not Implemented')

    def test_residuals_setter(self):
        self.skipTest('Not Implemented')

    def test_masked_datasets(self):
        self.skipTest('Not Implemented')

    def test_masked_datasets_setter(self):
        self.skipTest('Not Implemented')

    def test_best_ef_grid_point(self):
        self.skipTest('Not Implemented')

    def test_best_ef_grid_point_setter(self):
        self.skipTest('Not Implemented')

    def test_pspl_params(self):
        self.skipTest('Not Implemented')

    def test_pspl_params_setter(self):
        self.skipTest('Not Implemented')

    def test_best_af_grid_point(self):
        self.skipTest('Not Implemented')

    def test_best_af_grid_point_setter(self):
        self.skipTest('Not Implemented')

    def test_binary_params(self):
        self.skipTest('Not Implemented')

    def test_binary_params_setter(self):
        self.skipTest('Not Implemented')

    def test_results(self):
        self.skipTest('Not Implemented')

    def test_results_setter(self):
        self.skipTest('Not Implemented')

    def _make_fake_record(self, t_0, sigmas=None):
        #lightweight stand-in for a fit-result record
        return SimpleNamespace(
           params=SimpleNamespace(copy=lambda: {"t_0": t_0}),
            sigmas=sigmas or {},
        )

    def _make_fitter_for_exozippy(self, fit_type, all_fit_results):
        #adjust MMEXOFASTFitter to the actual class name/import path
        fitter = MMEXOFASTFitter.__new__(MMEXOFASTFitter)
        fitter.fit_type = fit_type
        fitter.all_fit_results = all_fit_results
        fitter.renorm_factors = {"OGLE": 1.0}
        fitter.mag_methods = []
        fitter.coords = None
        return fitter

    def test_initialize_exozippy(self):
        # t_0 below 2450000.0 shoul dbe baseline corrected in output
        key = "key1"
        record = self._make_fake_record(t_0=5000.0)
        fitter = self._make_fitter_for_exozippy(
            fit_type="point_lens",
            all_fit_results={key: record},
        )
        
        fitter._iter_parallax_point_lens_keys = lambda: [key]

        result = fitter.initialize_exozippy()

        self.assertEqual(len(result["fits"]), 1)
        self.assertAlmostEqual(
            result["fits"][0]["parameters"]["t_0"], 2455000.0
        )

        record_late = self._make_fake_record(t_0=24590000.0)
        fitter_late = self._make_fitter_for_exozippy(
            fit_type="point_lens",
            all_fit_results={key: record_late},

        )
        fitter_late._iter_parallax_point_lens_keys = lambda: [key]
        
        late_result = fitter_late.initialize_exozippy()
        #unsupported fit_type should raise
        fitter_bad = self._make_fitter_for_exozippy(
            fit_type="something_else",
            all_fit_results={},
        )

        fitter_bad._iter_parallax_point_lens_keys = lambda: [key]
        
        with self.assertRaises(NotImplementedError):
            fitter_bad.initialize_exozippy()
            
class TestSatelliteData(unittest.TestCase):

    def setUp(self):
        self.ground_data = None
        self.spitzer_data = None
        self.kepler_data = None
        self.skipTest('Need to setup test datasets for this test (among other things).')

    def do_test_file_list(self, files):
        self.skipTest('Not Implemented')
        fitter = MMEXOFASTFitter(files=files)
        assert fitter.n_loc == len(files)

    def _get_datasets(self, file_list):
        # Do something to read in the datasets
        return datasets

    def do_test_datasets(self, file_list):
        self.skipTest('Not Implemented')
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
