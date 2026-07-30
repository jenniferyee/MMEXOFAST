"""
Access to the 2018 WFIRST/Roman Microlensing Data Challenge dataset.

The data challenge light curves are not distributed with this package. They
are 132 MB, over PyPI's per-file limit, and the upstream repository

    https://github.com/microlensing-data-challenge/data-challenge-1

carries no license, so they are not ours to redistribute. They are instead
downloaded from upstream on first use and cached, by
:func:`fetch_light_curves`. The one file needed at package runtime, the W149
ephemerides, is handled separately by :func:`get_ephemerides`, which prefers
the byte-identical repository copy so that a checkout and CI never reach the
network for it.

Upstream names light curves ``ulwdc1_<num>_<band>.txt``, which does not match
the ``nYYYYMMDD.BAND.TELESCOPE`` convention that
:func:`mmexofast.observatories.get_kwargs` parses to identify a dataset's
telescope and bandpass. Fetched files are therefore renamed to

    n20180816.<band>.DC18.<num>.txt

The telescope tag is ``DC18``, not ``WFIRST18``, and that distinction is load
bearing. Upstream publishes magnitudes, and the ``DC18`` observatory is
registered with ``phot_fmt='mag'``; ``WFIRST18`` is registered with
``phot_fmt='flux'``. Tagging upstream files ``WFIRST18`` would tell
MulensModel that magnitudes are fluxes, silently corrupting the photometry.

Note that the ``WFIRST18``-tagged light curves under
``data/2018DataChallenge`` are not upstream files under a different name --
they are a 45 event subset converted to flux, exactly, with zero point 22.
Fetched files are left as magnitudes because that is upstream's native format
and MulensModel reads it directly, so no conversion is needed.
:func:`fetch_light_curves` always returns the upstream data rather than
substituting that subset for it, since the two differ in both sample size and
photometry format.

Every event is fetched, including the ones the answer key classes as
cataclysmic variables rather than microlensing. Those are part of the
challenge by design -- non-microlensing contaminants to test classification
-- and nothing in this package distinguishes them by filename.
"""

import os
import os.path
import re
import shutil
import tarfile

from .config import SAMPLE_DATA_PATH


DC18_DIRNAME = '2018DataChallenge'

_BASE_URL = ('https://raw.githubusercontent.com/microlensing-data-challenge/'
             'data-challenge-1/master/')

EPHEMERIDES_FILE = 'wfirst_ephemeris_W149.txt'
EPHEMERIDES_URL = _BASE_URL + EPHEMERIDES_FILE
LIGHT_CURVES_URL = _BASE_URL + 'lc.tar.gz'
EVENT_INFO_URL = _BASE_URL + 'event_info.txt'

# Nominal date stamped into every renamed light curve. The data challenge is a
# single simulated release, so this is a constant rather than a per-event
# observation date.
_NOMINAL_DATE = 'n20180816'

# Telescope tag for fetched files. DC18 is registered with phot_fmt='mag',
# matching the magnitudes upstream publishes. See the module docstring.
_TELESCOPE = 'DC18'

# Upstream light curve names, e.g. ulwdc1_004_W149.txt.
_UPSTREAM_RE = re.compile(r'^ulwdc1_(?P<num>\d+)_(?P<band>[^.]+)\.txt$')


def _sample_data_dir():
    """
    The repository copy of the data challenge, or None if not present.

    Returns
    -------
    str or None
    """
    if SAMPLE_DATA_PATH is None:
        return None

    candidate = os.path.join(SAMPLE_DATA_PATH, DC18_DIRNAME)
    return candidate if os.path.isdir(candidate) else None


def _cache_dir():
    """
    Directory holding the fetched dataset, created if necessary.

    Placed under astropy's cache directory, alongside the downloads made by
    ``astropy.utils.data.download_file``, so that clearing the astropy cache
    clears this too.

    Returns
    -------
    str
    """
    from astropy.config.paths import get_cache_dir

    path = os.path.join(get_cache_dir(), 'mmexofast', DC18_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def get_ephemerides():
    """
    Path to the W149 ephemerides file.

    Needed at runtime by the WFIRST18 and DC18 observatories. Prefers the
    repository copy so that a source checkout and CI never reach the network;
    otherwise downloads this one file (2.8 MB) and caches it. The full light
    curve set is not required for this.

    Resolved lazily, on first use of an observatory that needs it, so that
    importing mmexofast never performs network I/O.

    Returns
    -------
    str
        Path to the ephemerides file, local or cached.
    """
    local_dir = _sample_data_dir()
    if local_dir is not None:
        local = os.path.join(local_dir, EPHEMERIDES_FILE)
        if os.path.isfile(local):
            return local

    from astropy.utils.data import download_file

    return download_file(EPHEMERIDES_URL, cache=True)


def upstream_to_mmexofast_name(name):
    """
    Translate an upstream light curve filename to this package's convention.

    Parameters
    ----------
    name : str
        Upstream basename, e.g. ``'ulwdc1_004_W149.txt'``.

    Returns
    -------
    str or None
        Translated basename, e.g. ``'n20180816.W149.DC18.004.txt'``, or None
        if ``name`` is not an upstream light curve. The telescope tag is
        ``DC18`` because upstream publishes magnitudes; see the module
        docstring.
    """
    match = _UPSTREAM_RE.match(name)
    if match is None:
        return None

    return '{0}.{1}.{2}.{3:03d}.txt'.format(
        _NOMINAL_DATE, match.group('band'), _TELESCOPE,
        int(match.group('num')))


def _extract_and_rename(tarball, destination):
    """
    Unpack the light curve tarball into ``destination``, renaming as we go.

    Parameters
    ----------
    tarball : str
        Path to the downloaded ``lc.tar.gz``.
    destination : str
        Directory to write the renamed light curves into.

    Returns
    -------
    int
        Number of light curves written.
    """
    staging = os.path.join(destination, '_staging')
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    try:
        with tarfile.open(tarball) as tar:
            # filter='data' rejects absolute paths, parent traversal, and
            # special files. Required in 3.14, where it is the default, and
            # explicit here so 3.12 behaves identically.
            tar.extractall(staging, filter='data')

        written = 0
        # The archive unpacks to lc/, but locate the light curves by name
        # rather than assuming that layout.
        for root, _dirs, files in os.walk(staging):
            for name in files:
                new_name = upstream_to_mmexofast_name(name)
                if new_name is None:
                    continue

                shutil.move(os.path.join(root, name),
                            os.path.join(destination, new_name))
                written += 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return written


def fetch_light_curves(force=False):
    """
    Directory holding all 293 data challenge light curves, downloading once.

    Downloads ``lc.tar.gz`` (102 MB) and ``event_info.txt``, unpacks them,
    renames the light curves to this package's convention, and caches the
    result. Later calls reuse the cache.

    This always returns the upstream data, including in a source checkout.
    The ``WFIRST18`` files under ``data/2018DataChallenge`` are a 45 event
    flux-converted subset, not the same dataset under different names, so
    returning them here instead would silently change both the sample size
    and the photometry format. Use
    ``os.path.join(config.SAMPLE_DATA_PATH, '2018DataChallenge')`` to reach
    that subset.

    Parameters
    ----------
    force : bool, optional
        Re-download and re-unpack even if a cached copy exists. Has no effect
        when the repository copy is used.

    Returns
    -------
    str
        Directory containing light curves named
        ``nYYYYMMDD.BAND.DC18.NNN.txt`` (magnitudes), plus
        ``event_info.txt``.

    Notes
    -----
    The light curves are redistributed by neither this package nor PyPI; they
    are fetched from the upstream data challenge repository, which states no
    license. Cite the data challenge if you publish results based on them.
    """
    from astropy.utils.data import download_file

    destination = _cache_dir()
    marker = os.path.join(destination, '.complete')
    if os.path.isfile(marker) and not force:
        return destination

    event_info = download_file(EVENT_INFO_URL, cache=True)
    tarball = download_file(LIGHT_CURVES_URL, cache=True)

    written = _extract_and_rename(tarball, destination)
    if written == 0:
        raise RuntimeError(
            "No light curves found in {0!r}. The upstream archive layout may "
            "have changed; expected members named "
            "ulwdc1_<num>_<band>.txt.".format(LIGHT_CURVES_URL))

    shutil.copyfile(event_info, os.path.join(destination, 'event_info.txt'))

    with open(marker, 'w') as file_:
        file_.write('{0:d}\n'.format(written))

    return destination
