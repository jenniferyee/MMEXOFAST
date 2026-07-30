"""
Tests for the 2018 Data Challenge fetcher.

Nothing here touches the network. The 102 MB upstream archive is replaced by
a synthetic tarball with the same member naming, and ``download_file`` is
mocked, so extraction and renaming are exercised without a download.
"""

import os
import os.path
import shutil
import tarfile
import tempfile
import unittest
from unittest import mock

from mmexofast import config, dc18, observatories


def _make_upstream_tarball(
    path, events=(1, 4), bands=("W149", "Z087"), inside_lc_dir=True
):
    """
    Build a stand-in for the upstream lc.tar.gz.

    Members are named as upstream names them, optionally nested in an ``lc/``
    directory the way the real archive is.
    """
    staging = tempfile.mkdtemp()
    try:
        for num in events:
            for band in bands:
                name = "ulwdc1_{0:03d}_{1}.txt".format(num, band)
                with open(os.path.join(staging, name), "w") as file_:
                    file_.write("2458346.505461 21.026099 0.010545\n")

        with tarfile.open(path, "w:gz") as tar:
            for name in sorted(os.listdir(staging)):
                arcname = "lc/" + name if inside_lc_dir else name
                tar.add(os.path.join(staging, name), arcname=arcname)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


class TestUpstreamNameTranslation(unittest.TestCase):
    def test_translates_upstream_light_curve_names(self):
        cases = [
            ("ulwdc1_004_W149.txt", "n20180816.W149.DC18.004.txt"),
            ("ulwdc1_001_Z087.txt", "n20180816.Z087.DC18.001.txt"),
            ("ulwdc1_293_W149.txt", "n20180816.W149.DC18.293.txt"),
        ]
        for upstream, expected in cases:
            with self.subTest(upstream=upstream):
                self.assertEqual(
                    dc18.upstream_to_mmexofast_name(upstream), expected
                )

    def test_ignores_files_that_are_not_light_curves(self):
        for name in (
            "event_info.txt",
            "README.md",
            "wfirst_ephemeris_W149.txt",
            "master_file.txt",
        ):
            with self.subTest(name=name):
                self.assertIsNone(dc18.upstream_to_mmexofast_name(name))

    def test_translated_names_parse_back_to_telescope_and_band(self):
        """
        The whole point of renaming: the result must parse under the
        nYYYYMMDD.BAND.TELESCOPE convention.
        """
        name = dc18.upstream_to_mmexofast_name("ulwdc1_004_W149.txt")
        telescope, band = observatories.get_telescope_band_from_filename(name)
        self.assertEqual((telescope, band), ("DC18", "W149"))

    def test_translated_names_are_magnitudes_not_flux(self):
        """
        Upstream publishes magnitudes, so fetched files must resolve to the
        DC18 observatory (phot_fmt='mag'), never WFIRST18 (phot_fmt='flux').
        """
        name = dc18.upstream_to_mmexofast_name("ulwdc1_004_W149.txt")
        kwargs = observatories.get_kwargs(name)
        self.assertEqual(kwargs["phot_fmt"], "mag")


class TestExtractAndRename(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.tarball = os.path.join(self.tmp, "lc.tar.gz")
        self.destination = os.path.join(self.tmp, "out")
        os.makedirs(self.destination)

    def test_extracts_and_renames_every_light_curve(self):
        _make_upstream_tarball(self.tarball, events=(1, 4))
        written = dc18._extract_and_rename(self.tarball, self.destination)

        self.assertEqual(written, 4)
        self.assertEqual(
            sorted(os.listdir(self.destination)),
            [
                "n20180816.W149.DC18.001.txt",
                "n20180816.W149.DC18.004.txt",
                "n20180816.Z087.DC18.001.txt",
                "n20180816.Z087.DC18.004.txt",
            ],
        )

    def test_finds_light_curves_outside_the_lc_directory(self):
        """Layout is not assumed; members are located by name."""
        _make_upstream_tarball(self.tarball, events=(4,), inside_lc_dir=False)
        written = dc18._extract_and_rename(self.tarball, self.destination)

        self.assertEqual(written, 2)

    def test_removes_its_staging_directory(self):
        _make_upstream_tarball(self.tarball, events=(4,))
        dc18._extract_and_rename(self.tarball, self.destination)

        self.assertNotIn("_staging", os.listdir(self.destination))

    def test_content_is_preserved(self):
        _make_upstream_tarball(self.tarball, events=(4,), bands=("W149",))
        dc18._extract_and_rename(self.tarball, self.destination)

        written = os.path.join(self.destination, "n20180816.W149.DC18.004.txt")
        with open(written) as file_:
            self.assertEqual(
                file_.read(), "2458346.505461 21.026099 0.010545\n"
            )


class TestFetchLightCurves(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        self.cache = os.path.join(self.tmp, "cache")
        os.makedirs(self.cache)
        patcher = mock.patch.object(
            dc18, "_cache_dir", return_value=self.cache
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.tarball = os.path.join(self.tmp, "lc.tar.gz")
        _make_upstream_tarball(self.tarball, events=(1, 4))
        self.event_info = os.path.join(self.tmp, "event_info.txt")
        with open(self.event_info, "w") as file_:
            file_.write("ulwdc1_001 1 269.165 -29.0207\n")

    def _download(self, url, cache=True):
        if url == dc18.LIGHT_CURVES_URL:
            return self.tarball
        if url == dc18.EVENT_INFO_URL:
            return self.event_info
        raise AssertionError("unexpected URL: {0}".format(url))

    def test_fetches_unpacks_and_caches(self):
        with mock.patch(
            "astropy.utils.data.download_file", side_effect=self._download
        ) as download:
            result = dc18.fetch_light_curves()

        self.assertEqual(result, self.cache)
        self.assertIn("n20180816.W149.DC18.004.txt", os.listdir(result))
        self.assertIn("event_info.txt", os.listdir(result))
        self.assertEqual(download.call_count, 2)

    def test_second_call_reuses_the_cache(self):
        with mock.patch(
            "astropy.utils.data.download_file", side_effect=self._download
        ):
            dc18.fetch_light_curves()

        with mock.patch(
            "astropy.utils.data.download_file", side_effect=self._download
        ) as download:
            dc18.fetch_light_curves()

        download.assert_not_called()

    def test_force_redownloads(self):
        with mock.patch(
            "astropy.utils.data.download_file", side_effect=self._download
        ):
            dc18.fetch_light_curves()

        with mock.patch(
            "astropy.utils.data.download_file", side_effect=self._download
        ) as download:
            dc18.fetch_light_curves(force=True)

        self.assertEqual(download.call_count, 2)

    def test_does_not_substitute_the_repository_subset(self):
        """
        The WFIRST18 files in the repository are a flux-converted 45 event
        subset, so a source checkout must still fetch upstream rather than
        silently returning different data in a different photometry format.
        """
        if config.SAMPLE_DATA_PATH is None:
            self.skipTest("no sample data (installed package)")

        with mock.patch(
            "astropy.utils.data.download_file", side_effect=self._download
        ):
            result = dc18.fetch_light_curves()

        self.assertNotEqual(
            os.path.realpath(result),
            os.path.realpath(
                os.path.join(config.SAMPLE_DATA_PATH, dc18.DC18_DIRNAME)
            ),
        )

    def test_raises_if_the_archive_holds_no_light_curves(self):
        empty = os.path.join(self.tmp, "empty.tar.gz")
        with tarfile.open(empty, "w:gz") as tar:
            pass

        def download(url, cache=True):
            return empty if url == dc18.LIGHT_CURVES_URL else self.event_info

        with mock.patch(
            "astropy.utils.data.download_file", side_effect=download
        ):
            with self.assertRaises(RuntimeError) as ctx:
                dc18.fetch_light_curves()

        self.assertIn("ulwdc1", str(ctx.exception))


class TestGetEphemerides(unittest.TestCase):
    def test_prefers_the_repository_copy(self):
        """
        The repository copy is byte-identical to upstream, so a checkout and
        CI use it rather than downloading.
        """
        local_dir = dc18._sample_data_dir()
        if local_dir is None:
            self.skipTest("no sample data (installed package)")

        expected = os.path.join(local_dir, dc18.EPHEMERIDES_FILE)
        if not os.path.isfile(expected):
            self.skipTest("2018 Data Challenge data not present")

        with mock.patch("astropy.utils.data.download_file") as download:
            self.assertEqual(dc18.get_ephemerides(), expected)

        download.assert_not_called()

    def test_downloads_when_the_repository_copy_is_absent(self):
        with mock.patch.object(dc18, "SAMPLE_DATA_PATH", None):
            with mock.patch("astropy.utils.data.download_file") as download:
                download.return_value = "/cached/wfirst_ephemeris_W149.txt"
                result = dc18.get_ephemerides()

        self.assertEqual(result, "/cached/wfirst_ephemeris_W149.txt")
        download.assert_called_once_with(dc18.EPHEMERIDES_URL, cache=True)

    def test_importing_mmexofast_does_not_download(self):
        """
        The ephemerides is resolved lazily; importing the package, or building
        an observatory, must not perform network I/O.
        """
        with mock.patch("astropy.utils.data.download_file") as download:
            observatories.Observatory(
                name="Lazy", ephemerides_loader=dc18.get_ephemerides
            )

        download.assert_not_called()
