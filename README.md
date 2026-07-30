# MMEXOFAST

[![Tests](https://github.com/jenniferyee/MMEXOFAST/actions/workflows/tests.yml/badge.svg)](https://github.com/jenniferyee/MMEXOFAST/actions/workflows/tests.yml)

Automated fitting of microlensing light curves.

MMEXOFAST wraps [MulensModel](https://github.com/rpoleski/MulensModel) and
[sfit_minimizer](https://github.com/jenniferyee/sfit_minimizer) to run a staged
fitting workflow, from locating an event in the data through binary-lens
modeling, without requiring the parameters to be seeded by hand.

## Installation

```bash
pip install mmexofast
```

## Quick start

```python
import mmexofast as mmexo

fitter = mmexo.MMEXOFASTFitter(files=['n20180801.I.OGLE.dat'], fit_type='point_lens')
results = fitter.fit()
```

`fit()` returns an `AllFitResults` mapping, keyed by `FitKey`, holding every fit
the workflow produced. Records are also reachable by human-readable label:

```python
record = results['PSPL static']
print(record.params, record.sigmas, record.chi2())
```

`fit_type` is either `'point_lens'` or `'binary_lens'`. Instead of `files`, you
may pass `datasets=` with pre-built `MulensModel.MulensData` objects.

## The workflow

`MMEXOFASTFitter.fit()` builds and runs an ordered list of steps, grouped into
named stages:

| Stage | What it does |
|---|---|
| `event_search` | EventFinder grid (Kim et al. 2018) to locate `t_0` |
| `fit_static_point_lens` | Estimate point-lens parameters, then fit static PSPL (and FSPL if requested) |
| `fit_point_lens_parallax` | Fit the `u_0 > 0` and `u_0 < 0` parallax branches |
| `renormalize` | Rescale per-dataset errors, then refit |
| `search_for_anomaly` | AnomalyFinder grid over the residuals, then classify the anomaly |
| `fit_binary_lens` | Seed binary-lens parameters from the anomaly, then fit with emcee |
| `parallax_grids` | Grid search over the `(piE_E, piE_N)` plane |

Anomalies are classified as `'close'`, `'wide'`, or `'high_mag'`, which selects
which estimators seed the binary-lens fit.

Long runs can be stopped, inspected, and resumed:

```python
fitter = mmexo.MMEXOFASTFitter(
    files=files,
    fit_type='binary_lens',
    stop_after='search_for_anomaly',      # or 'stage:step', e.g. 'fit_binary_lens:fit_binary_lens_models'
    restart_file='event.pkl',             # checkpoint after every step
)
```

Re-running with the same `restart_file` picks up where the previous run stopped.

## Output

Plots, grid-search results, and results tables are written only if you ask for
them, via `OutputConfig`:

```python
from pathlib import Path

fitter = mmexo.MMEXOFASTFitter(
    files=files,
    fit_type='point_lens',
    output_config=mmexo.OutputConfig(
        output_dir=Path('results'),
        file_prefix='ob180383',
        save_plots=True,
        save_table=True,
    ),
)
```

## Data file naming

Filenames of the form `nYYYYMMDD.BAND.TELESCOPE.anything` are parsed to set the
bandpass, ephemerides, and plot properties for each dataset automatically. Pass
`datasets=` directly if your files don't follow that convention.

## Development

```bash
git clone https://github.com/jenniferyee/MMEXOFAST.git
cd MMEXOFAST
pip install -e '.[test]'

pytest source/mmexofast/unit_tests/            # everything
pytest source/mmexofast/unit_tests/ --fast     # skip grid searches and other slow tests
```

The sample data under `data/` and the scripts in `examples/` are part of the
repository, not the installed package; clone the repository to use them.

## Status

Alpha. The API may change between releases.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
