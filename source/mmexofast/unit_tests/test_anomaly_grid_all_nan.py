"""An AnomalyFinder grid that finds nothing must say so, not raise from numpy.

``AnomalyFinderGridSearch.do_fits`` leaves every chi2 column at NaN for any
grid window its data-sufficiency gate rejects (fewer than 5 good points, or no
successive coverage).  ``best`` then selects with ``np.nanargmax``, which is
correct -- but had no answer for the case where NOT ONE window was fittable,
and raised ``ValueError: All-NaN slice encountered`` from inside a property.

Four of the 44 DC2018 light curves die exactly that way.  Instrumenting their
grids showed the cause is not numerical: all 94375 grid points were rejected by
the gate.  The controls are what make that actionable -- a HEALTHY event has
only ~480-500 fittable points out of the same 94375, so ~99.5% NaN is the
normal state of this grid and must keep working.
"""

import unittest

import numpy as np

from mmexofast.gridsearches import AnomalyFinderGridSearch, NoAnomalyFoundError


class _Grid(AnomalyFinderGridSearch):
    """Bypass __init__ and the grid run: this tests selection, not fitting.

    grid_t_0/grid_t_eff are read-only properties that would lazily build a
    real grid, so they are overridden rather than assigned.
    """

    def __init__(self, results, n):
        self.results = results
        self._best = None
        self._anomalies = None
        self._n = n

    @property
    def grid_t_0(self):
        return np.arange(self._n, dtype=float)

    @property
    def grid_t_eff(self):
        return np.ones(self._n)


def _results(n, n_finite):
    """(n, 4) chi2 table with `n_finite` fittable rows, the rest all-NaN."""
    r = np.full((n, 4), np.nan)
    if n_finite:
        # chi2 columns [j1, j2, flat, zero]; zero > j1 makes dchi2_zero > 0.
        r[:n_finite, 0] = np.linspace(100.0, 90.0, n_finite)
        r[:n_finite, 1] = np.linspace(101.0, 91.0, n_finite)
        r[:n_finite, 2] = 150.0
        r[:n_finite, 3] = np.linspace(200.0, 300.0, n_finite)
    return r


class TestAnomalyGridAllNaN(unittest.TestCase):
    def test_mostly_nan_grid_still_selects_a_best_point(self):
        """The normal case: 0.5% fittable, as on a healthy DC2018 event."""
        # Arrange
        grid = _Grid(_results(94375, 500), 94375)

        # Act
        best = grid.best

        # Assert
        assert best is not None
        assert np.isfinite(best["t_0"])
        assert np.isfinite(best["dchi2_zero"])

    def test_all_nan_grid_returns_none_instead_of_raising(self):
        """
        Given a grid on which every window was rejected,
        When best is read,
        Then it is None rather than a ValueError out of nanargmax.
        """
        # Arrange
        grid = _Grid(_results(94375, 0), 94375)

        # Act / Assert
        assert grid.best is None

    def test_no_results_at_all_is_still_none(self):
        """The pre-existing contract for "no answer" is unchanged."""
        assert _Grid(None, 0).best is None

    def test_the_named_error_is_catchable_as_valueerror(self):
        """
        Given code written against the old behaviour,
        When it catches ValueError,
        Then NoAnomalyFoundError is still caught.

        The compatibility that lets this ship without breaking callers: the
        condition used to surface as ValueError from numpy, so the new type
        subclasses it rather than replacing it.
        """
        assert issubclass(NoAnomalyFoundError, ValueError)
        try:
            raise NoAnomalyFoundError("no anomaly")
        except ValueError as exc:
            assert "no anomaly" in str(exc)


if __name__ == "__main__":
    unittest.main()
