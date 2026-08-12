"""An emcee polish must never return a point worse than its seed.

Both halves of the guarantee are pinned here:

1. ``make_starting_vector`` puts the UNPERTURBED seed in the ensemble as
   walker 0, so the seed is a live member the stretch move competes against
   rather than a point the ensemble has never visited.
2. ``run`` selects its best-fit point over the full chain (burn-in included)
   AND the starting ensemble, so even a run whose walkers all drift downhill
   reports the seed back rather than wherever they ended up.

The motivating failure was 2018 Data Challenge event 128: seeded at
chi2 37988, the binary-lens polish returned a "fit" at 147227, because the
best-fit point was selected from the post-burn-in chain only.
"""

import unittest

import numpy as np

from mmexofast import fitters


class _StubFitter(fitters.EmceeLCFitter):
    """EmceeLCFitter with the model replaced by an analytic log-probability.

    Subclassed rather than mocked so the real make_starting_vector and the
    real run() selection logic are what get exercised; only the likelihood
    and the event bookkeeping are stubbed out.
    """

    def __init__(self, ln_prob_fn, **kwargs):
        super().__init__(**kwargs)
        self._ln_prob_fn = ln_prob_fn
        self._event = object()  # non-None: skip initialize_event()

    def ln_prob(self, theta):
        return float(self._ln_prob_fn(np.asarray(theta, dtype=float)))

    # run() sets `self.event = best_theta` then reads model parameters off
    # it; capture the vector instead of building a MulensModel.Event.
    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, theta):
        self._recorded_theta = np.asarray(theta, dtype=float)
        self._event = _StubEvent(self._recorded_theta, self._ln_prob_fn)


class _StubEvent:
    def __init__(self, theta, ln_prob_fn):
        self.model = _StubModel(theta)
        self._chi2 = -2.0 * ln_prob_fn(theta)

    def get_chi2(self):
        return self._chi2


class _StubModel:
    def __init__(self, theta):
        self.parameters = _StubParameters(theta)


class _StubParameters:
    def __init__(self, theta):
        self.parameters = {"t_0": float(theta[0]), "u_0": float(theta[1])}


def _make_fitter(ln_prob_fn, n_walkers=8, n_steps=30, n_burn=10):
    return _StubFitter(
        ln_prob_fn=ln_prob_fn,
        initial_guess={"t_0": 0.0, "u_0": 0.0},
        sigmas={"t_0": 1.0, "u_0": 1.0},
        parameters_to_fit=["t_0", "u_0"],
        emcee_settings={
            "n_walkers": n_walkers,
            "n_steps": n_steps,
            "n_burn": n_burn,
            "n_dim": 2,
            "acceptance_fraction": None,
        },
    )


class TestSeedIsInEnsemble(unittest.TestCase):
    def test_walker_zero_is_the_unperturbed_seed(self):
        """Given sigmas, when the ensemble is built, walker 0 is the seed."""
        fitter = _make_fitter(lambda th: -np.sum(th**2))
        np.random.seed(12345)
        vectors = fitter.make_starting_vector()

        np.testing.assert_allclose(vectors[0], [0.0, 0.0], atol=0.0)

    def test_other_walkers_are_perturbed(self):
        """Only walker 0 is pinned; the rest still scatter."""
        fitter = _make_fitter(lambda th: -np.sum(th**2))
        np.random.seed(12345)
        vectors = np.asarray(fitter.make_starting_vector())

        assert np.all(np.any(vectors[1:] != 0.0, axis=1))
        assert np.std(vectors[1:]) > 0.1


class TestBestIsNeverWorseThanSeed(unittest.TestCase):
    def test_downhill_run_returns_the_seed(self):
        """A likelihood that rewards fleeing the seed must not lose it.

        The target puts a narrow spike at the origin (the seed) on top of a
        broad bowl centred far away, so every walker is driven off the spike
        and the post-burn-in chain contains nothing near it.  Selecting the
        best over post-burn-in samples alone therefore returned a point far
        worse than the input; the fix must return the seed.
        """
        far = np.array([50.0, 50.0])

        def ln_prob(theta):
            spike = 1000.0 if np.all(np.abs(theta) < 1e-9) else 0.0
            return spike - 0.01 * np.sum((theta - far) ** 2)

        fitter = _make_fitter(ln_prob, n_walkers=8, n_steps=40, n_burn=20)
        np.random.seed(7)
        msg = fitter.run()

        assert msg is None
        seed_lp = ln_prob(np.array([0.0, 0.0]))
        best_lp = ln_prob(fitter.best_theta)
        assert best_lp >= seed_lp, (
            f"polish returned lp {best_lp} against a seed at {seed_lp}"
        )
        np.testing.assert_allclose(fitter.best_theta, [0.0, 0.0], atol=1e-9)

    def test_best_can_come_from_burn_in(self):
        """The optimum visited during burn-in must not be discarded."""
        fitter = _make_fitter(
            lambda th: -np.sum(th**2), n_walkers=8, n_steps=30, n_burn=25
        )
        np.random.seed(3)
        msg = fitter.run()

        assert msg is None
        chain = fitter.sampler.chain.reshape((-1, 2))
        prob = fitter.sampler.lnprobability.reshape((-1))
        best_anywhere = max(
            float(np.max(prob)), -float(np.sum(np.zeros(2) ** 2))
        )
        np.testing.assert_allclose(
            -np.sum(fitter.best_theta**2), best_anywhere, rtol=0, atol=1e-9
        )
        assert len(chain) == fitter.sampler.iteration * 8

    def test_improving_run_still_reports_the_improvement(self):
        """The guard must not clamp a genuinely better result to the seed."""
        target = np.array([2.0, -1.0])
        fitter = _make_fitter(
            lambda th: -np.sum((th - target) ** 2),
            n_walkers=16,
            n_steps=200,
            n_burn=50,
        )
        np.random.seed(11)
        msg = fitter.run()

        assert msg is None
        seed_lp = -float(np.sum(target**2))
        best_lp = -float(np.sum((fitter.best_theta - target) ** 2))
        assert best_lp > seed_lp


if __name__ == "__main__":
    unittest.main()
