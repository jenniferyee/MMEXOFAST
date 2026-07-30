import glob
import os
import unittest

import MulensModel
import pytest

from mmexofast.config import DATA_PATH
from mmexofast.mulens_object_config import EventConfig, ModelConfig
from mmexofast.observatories import get_kwargs

# ===========================================================================
# Module-level constants
# ===========================================================================

PSPL_PARAMS = {"t_0": 2460000.0, "u_0": 0.1, "t_E": 30.0, "rho": 0.001}
COORDS = "18:00:00 -30:00:00"
BANDPASS = "I"
GAMMA = 0.5
U_COEFF = 0.3
MAG_METHODS = [2460000.0, "finite_source_LD_WittMao94", 2460100.0]
DEFAULT_MAG_METHOD = "finite_source_uniform_Gould94"
FIX_BLEND_FLUX_VALUE = 0.0
DATA_REF = 1

OB05390_FILES = sorted(
    glob.glob(os.path.join(DATA_PATH, "OB05390", "n200*.txt"))
)

with open(os.path.join(DATA_PATH, "OB05390", "coords.txt")) as f:
    OB05390_COORDS = f.read().strip()


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def datasets():
    return [
        MulensModel.MulensData(file_name=f, **get_kwargs(f))
        for f in OB05390_FILES
    ]


# ===========================================================================
# TestModelConfig
# ===========================================================================


class TestModelConfig:
    """
    Tests that ModelConfig correctly constructs and configures
    MulensModel.Model objects.

    Each test checks one field stored in ModelConfig or one argument
    passed to build(), verifying that the resulting Model reflects the
    supplied value.

    Tests
    -----
    test_default_config_builds_model        — empty ModelConfig builds a valid Model
    test_parameters_applied                 — parameters passed to build() appear on model
    test_coords_passed_to_model             — coords stored in config reaches model
    test_limb_coeff_gamma_applied           — gamma limb darkening applied to model
    test_limb_coeff_u_applied               — u limb darkening applied to model
    test_magnification_methods_applied      — mag methods passed to build() applied to model
    test_default_magnification_method_applied — default mag method passed to build() applied
    """

    def test_default_config_builds_model(self):
        """Empty ModelConfig builds a valid Model without error."""
        config = ModelConfig()
        model = config.build(parameters=PSPL_PARAMS)
        assert isinstance(model, MulensModel.Model)

    def test_parameters_applied(self):
        """Parameters passed to build() appear on the resulting model."""
        config = ModelConfig()
        model = config.build(parameters=PSPL_PARAMS)
        assert model.parameters.t_0 == PSPL_PARAMS["t_0"]
        assert model.parameters.u_0 == PSPL_PARAMS["u_0"]
        assert model.parameters.t_E == PSPL_PARAMS["t_E"]

    def test_coords_passed_to_model(self):
        """coords stored in ModelConfig is passed to the model constructor."""
        config = ModelConfig(coords=COORDS)
        model = config.build(parameters=PSPL_PARAMS)
        assert model.coords is not None

    def test_limb_coeff_gamma_applied(self):
        """limb_coeff_gamma stored in ModelConfig is applied to the model."""
        config = ModelConfig(limb_coeff_gamma={BANDPASS: GAMMA})
        model = config.build(parameters=PSPL_PARAMS)
        assert model.get_limb_coeff_gamma(BANDPASS) == GAMMA

    def test_limb_coeff_u_applied(self):
        """limb_coeff_u stored in ModelConfig is applied to the model."""
        config = ModelConfig(limb_coeff_u={BANDPASS: U_COEFF})
        model = config.build(parameters=PSPL_PARAMS)
        assert model.get_limb_coeff_u(BANDPASS) == U_COEFF

    def test_magnification_methods_applied(self):
        """magnification_methods passed to build() is applied to the model."""
        config = ModelConfig()
        model = config.build(
            parameters=PSPL_PARAMS,
            magnification_methods=MAG_METHODS,
        )
        assert model.methods == MAG_METHODS

    def test_default_magnification_method_applied(self):
        """default_magnification_method passed to build() is applied to the model."""
        config = ModelConfig()
        model = config.build(
            parameters=PSPL_PARAMS,
            default_magnification_method=DEFAULT_MAG_METHOD,
        )
        assert model.default_magnification_method == DEFAULT_MAG_METHOD

    def test_magnification_methods_parameters_survive_repeated_builds(self):
        """One parameters dict can build many models.

        MulensModel hands the per-method dict straight to the magnification
        objects and injects a 'trajectory' key into it, so a caller that
        holds one dict and builds repeatedly (MulensFitter across an emcee
        run) used to have its dict poisoned by the first model and rejected
        by the second. build() must copy.
        """
        binary_params = {
            "t_0": 2458271.6,
            "u_0": 0.91,
            "t_E": 27.7,
            "s": 1.64,
            "alpha": 35.9,
            "rho": 0.30,
            "q": 3.4e-3,
        }
        methods = [2458182.7, "VBBL", 2458310.6]
        methods_parameters = {"VBBL": {"accuracy": 0.01}}
        times = [2458270.0, 2458271.0, 2458272.0]

        config = ModelConfig()
        for _ in range(3):
            model = config.build(
                parameters=binary_params,
                magnification_methods=methods,
                magnification_methods_parameters=methods_parameters,
            )
            # Forces the magnification objects to be built, which is what
            # injects 'trajectory'.
            model.get_magnification(times)

        assert methods_parameters == {"VBBL": {"accuracy": 0.01}}


# ===========================================================================
# TestEventConfig
# ===========================================================================


class TestEventConfig:
    """
    Tests that EventConfig correctly constructs MulensModel.Event objects.

    Each test checks one field stored in EventConfig, verifying that the
    resulting Event reflects the supplied value.

    Tests
    -----
    test_default_config_builds_event  — empty EventConfig builds a valid Event
    test_coords_passed_to_event       — coords stored in config reaches event
    test_fix_blend_flux_applied       — blend flux fixed to 0.0 after fit_fluxes()
    test_fix_source_flux_applied      — source flux fixed after fit_fluxes()
    test_data_ref_applied             — data_ref stored in config reaches event
    """

    def test_default_config_builds_event(self, datasets):
        """Empty EventConfig builds a valid Event without error."""
        config = EventConfig()
        model = ModelConfig().build(parameters=PSPL_PARAMS)
        event = config.build(model=model, datasets=datasets)
        assert isinstance(event, MulensModel.Event)

    def test_coords_passed_to_event(self, datasets):
        """coords stored in EventConfig is passed to the event constructor."""
        config = EventConfig(coords=COORDS)
        model = ModelConfig().build(parameters=PSPL_PARAMS)
        event = config.build(model=model, datasets=datasets)
        assert event.coords is not None

    def test_fix_blend_flux_applied(self, datasets):
        """
        fix_blend_flux stored in EventConfig is applied: blend_flux is
        fixed to 0.0 after fit_fluxes().
        """
        fix_blend_flux = {datasets[0]: FIX_BLEND_FLUX_VALUE}
        config = EventConfig(fix_blend_flux=fix_blend_flux)
        model = ModelConfig().build(parameters=PSPL_PARAMS)
        event = config.build(model=model, datasets=datasets)
        event.fit_fluxes()
        assert event.fits[0].blend_flux == FIX_BLEND_FLUX_VALUE

    def test_fix_source_flux_applied(self, datasets):
        """
        fix_source_flux stored in EventConfig is applied: source_flux
        is fixed after fit_fluxes().

        Uses the source flux from an initial unconstrained fit as the
        fixed value, so the value is self-consistent with the data.
        """
        # Get a reasonable source flux value from an unconstrained fit
        model = ModelConfig().build(parameters=PSPL_PARAMS)
        ref_event = EventConfig().build(model=model, datasets=datasets)
        ref_event.fit_fluxes()
        source_flux_value = ref_event.fits[0].source_fluxes[0]

        # Fix to that value and verify
        fix_source_flux = {datasets[0]: source_flux_value}
        config = EventConfig(fix_source_flux=fix_source_flux)
        model = ModelConfig().build(parameters=PSPL_PARAMS)
        event = config.build(model=model, datasets=datasets)
        event.fit_fluxes()
        assert event.fits[0].source_fluxes[0] == source_flux_value

    @unittest.skip(
        "There's some problem with how MulensModel.Event stores data_ref."
    )
    def test_data_ref_applied(self, datasets):
        """data_ref stored in EventConfig is passed to the event constructor."""
        config = EventConfig(data_ref=DATA_REF)
        model = ModelConfig().build(parameters=PSPL_PARAMS)
        event = config.build(model=model, datasets=datasets)

        assert event.data_ref == DATA_REF

    def test_mag_methods_applied_to_model_in_event(self, datasets):
        """
        magnification_methods passed to ModelConfig.build() are applied to
        the model inside the resulting Event.
        """
        model = ModelConfig().build(
            parameters=PSPL_PARAMS,
            magnification_methods=MAG_METHODS,
        )
        config = EventConfig()
        event = config.build(model=model, datasets=datasets)
        assert event.model.methods == MAG_METHODS

    def test_limb_coeff_gamma_applied_to_model_in_event(self, datasets):
        """
        limb_coeff_gamma stored in ModelConfig is applied to the model
        inside the resulting Event.
        """
        model = ModelConfig(limb_coeff_gamma={BANDPASS: GAMMA}).build(
            parameters=PSPL_PARAMS,
        )
        config = EventConfig()
        event = config.build(model=model, datasets=datasets)
        assert event.model.get_limb_coeff_gamma(BANDPASS) == GAMMA

    def test_limb_coeff_u_applied_to_model_in_event(self, datasets):
        """
        limb_coeff_u stored in ModelConfig is applied to the model
        inside the resulting Event.
        """
        model = ModelConfig(limb_coeff_u={BANDPASS: U_COEFF}).build(
            parameters=PSPL_PARAMS,
        )
        config = EventConfig()
        event = config.build(model=model, datasets=datasets)
        assert event.model.get_limb_coeff_u(BANDPASS) == U_COEFF
