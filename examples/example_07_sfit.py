"""
Example to show that the basic sfit fit works as expected.

Truth values:
u0        alpha       t0         tE        rE       thetaE    piE     rhos
-1.52343 38.6665 1790.31690896 6.6877 2.15354 0.35275 0.171343 0.00120717
"""
import matplotlib.pyplot as plt

import MulensModel
import mmexofast as mmexo
from data_for_test_examples import datasets

results_EF = {'t_0': 2460023.2844586717, 't_eff': 9.988721231519582,
              'j': 2, 'chi2': -42226.356330652685}

est_params = mmexo.estimate_params.get_PSPL_params(results_EF, datasets)
fitter = mmexo.fitters.SFitFitter(
    datasets=datasets, initial_model_params=est_params, verbose=True)
fitter.run()
best = fitter.best
best.pop('chi2')

print('estimated parameters', est_params)
init_event = MulensModel.Event(
    datasets=datasets, model=MulensModel.Model(est_params))
print('initial chi2', init_event.get_chi2())
init_event.plot(title='Initial Parameters')

event = MulensModel.Event(datasets=datasets, model=MulensModel.Model(best))
print('final parameters', event.model.parameters.parameters)
print(event)
print('final chi2', event.get_chi2())
event.plot(title='Final Parameters')

plt.show()
