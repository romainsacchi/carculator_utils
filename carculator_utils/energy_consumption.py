"""
energy_consumption.py contains the class EnergyConsumption Model
which exposes two methods:
one for calculating the auxiliary energy needs,
and another one for calculating the motive
energy needs.
"""

import csv
from typing import Any, List, Tuple, Union

import numexpr as ne
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from numpy import ndarray
from xarray import DataArray

from . import DATA_DIR
from .driving_cycles import (
    get_driving_cycle_specs,
    get_standard_driving_cycle_and_gradient,
)

MONTHLY_AVG_TEMP = "monthly_avg_temp.csv"


def _(obj: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """Add a trailing dimension to make input arrays broadcast correctly"""
    if isinstance(obj, (np.ndarray, xr.DataArray)):
        return np.expand_dims(obj, -1)

    return obj


def __(obj: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """Add a heading dimension to make input arrays broadcast correctly"""
    if isinstance(obj, (np.ndarray, xr.DataArray)):
        return np.expand_dims(obj, 0)

    return obj


def get_default_driving_cycle_name(vehicle_type) -> str:
    """Get the default driving cycle name"""
    return list(get_driving_cycle_specs()["columns"][vehicle_type].keys())[0]


def get_efficiency_coefficients(vehicle_type: str) -> [Any, None]:
    # load yaml file to retrieve efficiency coefficients

    # if file does not exist, return None
    if not (DATA_DIR / "efficiency" / f"{vehicle_type}.yaml").exists():
        return None

    with open(DATA_DIR / "efficiency" / f"{vehicle_type}.yaml") as f:
        efficiency_coefficients = yaml.load(f, Loader=yaml.FullLoader)

    return efficiency_coefficients


def get_country_temperature(country):
    """
    Retrieves mothly average temperature
    :type country: country for which to retrieve temperature values
    :return:
    """

    with open(DATA_DIR / MONTHLY_AVG_TEMP) as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if row[2] == country:
                return np.array([float(i) for i in row[3:]])

    print(
        f"Could not find monthly average temperature series for {country}. "
        f"Uses those for CH instead."
    )

    with open(DATA_DIR / MONTHLY_AVG_TEMP) as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if row[2] == "CH":
                return np.array([int(i) for i in row[3:]])


def convert_to_xr(data):
    return xr.DataArray(
        data,
        dims=["second", "value", "year", "powertrain", "size", "parameter"],
        coords={
            "second": range(0, data.shape[0]),
            "value": range(0, data.shape[1]),
            "year": range(0, data.shape[2]),
            "powertrain": range(0, data.shape[3]),
            "size": range(0, data.shape[4]),
            "parameter": [
                "rolling resistance",
                "air resistance",
                "gradient resistance",
                "kinetic energy",
                "motive energy at wheels",
                "motive energy",
                "negative motive energy",
                "recuperated energy",
                "auxiliary energy",
                "cooling energy",
                "heating energy",
                "battery cooling energy",
                "battery heating energy",
                "power load",
                "transmission efficiency",
                "engine efficiency",
                "velocity",
            ],
        },
    )


class EnergyConsumptionModel:
    """
    Calculate energy consumption of a vehicle for a
    given driving cycle and vehicle parameters.

    Based on a selected driving cycle, this class calculates
    the acceleration needed and provides
    two methods:

        - :func:`~energy_consumption.EnergyConsumptionModel.aux_energy_per_km` calculates the energy needed to power auxiliary services
        - :func:`~energy_consumption.EnergyConsumptionModel.motive_energy_per_km` calculates the energy needed to move the vehicle over 1 km

    Acceleration is calculated as the difference between velocity
    at t_2 and velocity at t_0, divided by 2.
    See for example: http://www.unece.org/fileadmin/DAM/trans/doc/2012/wp29grpe/WLTP-DHC-12-07e.xls

    :param cycle: Driving cycle. Pandas Series of second-by-second speeds (km/h) or name (str)
    :type cycle: np.ndarray
    :param rho_air: Mass per unit volume of air. Set to (1.225 kg/m3) by default.
    :type rho_air: float
    :param gradient: Road gradient per second of driving, in degrees.
    None by default. Should be passed as an array of length equal
    to the length of the driving cycle.
    :type gradient: numpy.ndarray

    :ivar rho_air: Mass per unit volume of air. Value of 1.204 at 23C (test temperature for WLTC).
    :vartype rho_air: float
    :ivar velocity: Time series of speed values, in meters per second.
    :vartype velocity: numpy.ndarray
    :ivar acceleration: Time series of acceleration, calculated as
    increment in velocity per interval of 1 second, in meter per second^2.
    :vartype acceleration: numpy.ndarray

    """

    def __init__(
        self,
        vehicle_type: str,
        vehicle_size: List[str],
        powertrains: List[str],
        cycle: Union[str, np.ndarray],
        gradient: Union[str, np.ndarray],
        rho_air: float = 1.204,
        country: str = "CH",
        ambient_temperature: Union[float, np.ndarray] = None,
        indoor_temperature: Union[float, np.ndarray] = 20,
    ) -> None:
        if not isinstance(vehicle_size, list):
            vehicle_size = [vehicle_size]

        self.rho_air = rho_air

        if isinstance(cycle, np.ndarray):
            self.cycle_name = "custom"
            self.cycle = cycle.reshape(-1, 1)

            if gradient is not None:
                self.gradient = gradient.reshape(-1, 1)
            else:
                self.gradient = np.zeros_like(self.cycle)

        else:
            self.cycle_name = cycle
            self.cycle, self.gradient = get_standard_driving_cycle_and_gradient(
                vehicle_type, vehicle_size, self.cycle_name
            )

        self.country = country
        self.vehicle_type = vehicle_type
        self.vehicle_size = vehicle_size
        self.powertrains = powertrains

        if "Micro" in vehicle_size:
            idx = vehicle_size.index("Micro")
            self.cycle[:, idx] = np.clip(self.cycle[:, idx], 0, 90)

        assert len(self.cycle) == len(
            self.gradient
        ), "The length of the driving cycle and the gradient must be the same."

        # Unit conversion km/h to m/s
        self.velocity = np.where(np.isnan(self.cycle), 0, (self.cycle * 1000) / 3600)
        self.velocity = self.velocity[:, None, None, None, :]
        self.driving_time = self.find_last_driving_second()

        # Model acceleration as difference in velocity between
        # time steps (1 second)
        # Zero at first value
        self.acceleration = np.zeros_like(self.velocity)
        self.acceleration[1:-1] = (self.velocity[2:, ...] - self.velocity[:-2, ...]) / 2

        self.efficiency_coefficients = get_efficiency_coefficients(vehicle_type)

        self.ambient_temperature = ambient_temperature
        self.indoor_temperature = indoor_temperature

    def calculate_hvac_energy(
        self,
        hvac_power,
        battery_cooling_unit,
        battery_heating_unit,
    ) -> tuple[Any, Any, Any, Any]:
        if self.ambient_temperature is not None:
            if isinstance(self.ambient_temperature, (float, int)):
                self.ambient_temperature = np.resize(self.ambient_temperature, (12,))
            else:
                self.ambient_temperature = np.array(self.ambient_temperature)
                assert (
                    len(self.ambient_temperature) == 12
                ), "Ambient temperature must be a 12-month array"
        else:
            self.ambient_temperature = np.resize(
                get_country_temperature(self.country), (12,)
            )

        if self.indoor_temperature is not None:
            if isinstance(self.indoor_temperature, (float, int)):
                self.indoor_temperature = np.resize(self.indoor_temperature, (12,))
            else:
                self.indoor_temperature = np.array(self.indoor_temperature)
                assert (
                    len(self.indoor_temperature) == 12
                ), "Indoor temperature must be a 12-month array"

        # use ambient temperature if provided, otherwise
        # monthly temperature average (12 values)
        # relation between ambient temperature
        # and HVAC power required
        # from https://doi.org/10.1016/j.energy.2018.12.064
        amb_temp_data_points = np.array([-30, -20, -10, 0, 10, 20, 30, 40])
        pct_power_HVAC = np.array([0.95, 0.54, 0.29, 0.13, 0.04, 0.08, 0.45, 0.7])

        # Heating power as long as ambient temperature, in W
        # is below the comfort indoor temperature
        p_heating = (
            np.where(
                self.ambient_temperature < self.indoor_temperature,
                np.interp(
                    self.ambient_temperature, amb_temp_data_points, pct_power_HVAC
                ),
                0,
            ).mean()
            * hvac_power
        ).values

        # Cooling power as long as ambient temperature, in W
        # is above the comfort indoor temperature
        p_cooling = (
            np.where(
                self.ambient_temperature >= self.indoor_temperature,
                np.interp(
                    self.ambient_temperature, amb_temp_data_points, pct_power_HVAC
                ),
                0,
            ).mean()
            * hvac_power
        ).values

        # We want to add power draw for battery cooling
        # and battery heating

        # battery cooling occurring above 20C, in W
        p_battery_cooling = np.where(
            self.ambient_temperature > 20, _(battery_cooling_unit), 0
        )
        p_battery_cooling = p_battery_cooling.mean(-1)

        # battery heating occurring below 5C, in W
        p_battery_heating = np.where(
            self.ambient_temperature < 5, _(battery_heating_unit), 0
        )
        p_battery_heating = p_battery_heating.mean(-1)

        return p_cooling, p_heating, p_battery_cooling, p_battery_heating

    def find_last_driving_second(self) -> ndarray:
        """
        Find the last second of the driving cycle that is not zero.
        """

        # find last index where velocity is greater than 0
        # along the last axis of self.velocity
        # let's iterate through the last axis of self.velocity
        # and find the last index where velocity is greater than 0
        driving_time = np.zeros_like(self.velocity)

        for i in range(self.velocity.shape[-1]):
            last_index = np.where(self.velocity[..., i] > 0)[0][-1]
            driving_time[:last_index, ..., i] = 1

        return driving_time

    def aux_energy_per_km(
        self,
        aux_power: Union[xr.DataArray, np.array],
        efficiency: Union[xr.DataArray, np.array],
        hvac_power: Union[xr.DataArray, np.array] = None,
        battery_cooling_unit: Union[xr.DataArray, np.array] = None,
        battery_heating_unit: Union[xr.DataArray, np.array] = None,
        heat_pump_cop_cooling: Union[xr.DataArray, np.array] = None,
        heat_pump_cop_heating: Union[xr.DataArray, np.array] = None,
        cooling_consumption: Union[xr.DataArray, np.array] = None,
        heating_consumption: Union[xr.DataArray, np.array] = None,
    ) -> Union[tuple[Any, Any, Any, Any, Any], Any]:
        """
        Calculate energy used other than motive energy per km driven.

        :param aux_power: Total power needed for auxiliaries, heating, and cooling (W)
        :param efficiency: Efficiency of electricity generation (dimensionless, between 0.0 and 1.0).
                Battery electric vehicles should have efficiencies of one here, as we account for
                battery efficiencies elsewhere.
        :returns: total auxiliary energy in kJ/km

        """

        _o = lambda x: np.where(x == 0, 1, x)

        if hvac_power is not None:
            (
                p_cooling,
                p_heating,
                p_battery_cooling,
                p_battery_heating,
            ) = self.calculate_hvac_energy(
                hvac_power=hvac_power,
                battery_cooling_unit=battery_cooling_unit,
                battery_heating_unit=battery_heating_unit,
            )

            return (
                aux_power.T.values * np.where(self.velocity > 0, 1, 0),
                (p_cooling / _o(heat_pump_cop_cooling) * cooling_consumption).T.values
                * self.driving_time,
                (p_heating / _o(heat_pump_cop_heating) * heating_consumption).T.values
                * self.driving_time,
                p_battery_cooling.T * self.driving_time,
                p_battery_heating.T * self.driving_time,
            )

        _c = lambda x: x.values if isinstance(x, xr.DataArray) else x

        # Provide energy in kJ / km (1 J = 1 Ws)
        auxiliary_energy = (
            _c(aux_power).T[None, ...] * 1000 / 1000  # Watts  # m / km  # 1 / (J / kJ)
        )

        efficiency = _c(efficiency)

        return auxiliary_energy / _o(efficiency)

    def calculate_efficiency(
        self,
        efficiency: xr.DataArray,
        engine_load: xr.DataArray,
        efficiency_type: str,
    ) -> xr.DataArray:
        if self.efficiency_coefficients is None:
            return xr.where(efficiency == 0, 1, efficiency)

        # Calculate efficiency based on engine load
        pwts = {
            "ICEV-p": "gasoline",
            "ICEV-d": "diesel",
            "HEV-p": "gasoline",
            "HEV-d": "diesel",
            "ICEV-g": "compressed gas",
            "PHEV-c-p": "gasoline",
            "PHEV-c-d": "diesel",
            "BEV": "electric",
            "BEV-depot": "electric",
            "BEV-opp": "electric",
            "BEV-motion": "electric",
            "PHEV-e": "electric",
            "FCEV": "electric",
        }

        # Calculate efficiency based on engine load
        for p, pwt in enumerate(self.powertrains):
            if pwt not in pwts:
                continue

            efficiency[:, :, :, p] = np.clip(
                np.interp(
                    engine_load[:, :, :, p],
                    np.fromiter(
                        self.efficiency_coefficients[pwts[pwt]][efficiency_type].keys(),
                        dtype=float,
                    ),
                    np.fromiter(
                        self.efficiency_coefficients[pwts[pwt]][
                            efficiency_type
                        ].values(),
                        dtype=float,
                    ),
                ),
                0.00,
                1.0,
            )

        return efficiency

    def motive_energy_per_km(
        self,
        driving_mass: Union[xr.DataArray, np.array],
        rr_coef: Union[xr.DataArray, np.array],
        drag_coef: Union[xr.DataArray, np.array],
        frontal_area: Union[xr.DataArray, np.array],
        electric_motor_power: Union[xr.DataArray, np.array],
        engine_power: Union[xr.DataArray, np.array],
        recuperation_efficiency: Union[xr.DataArray, np.array],
        aux_power: Union[xr.DataArray, np.array],
        battery_charge_eff: Union[xr.DataArray, np.array],
        battery_discharge_eff: Union[xr.DataArray, np.array],
        engine_efficiency: Union[xr.DataArray, np.array, None] = None,
        transmission_efficiency: Union[xr.DataArray, np.array, None] = None,
        fuel_cell_system_efficiency: Union[xr.DataArray, np.array] = None,
        hvac_power: Union[xr.DataArray, np.array] = None,
        battery_cooling_unit: Union[xr.DataArray, np.array] = None,
        battery_heating_unit: Union[xr.DataArray, np.array] = None,
        heat_pump_cop_cooling: Union[xr.DataArray, np.array] = None,
        heat_pump_cop_heating: Union[xr.DataArray, np.array] = None,
        cooling_consumption: Union[xr.DataArray, np.array] = None,
        heating_consumption: Union[xr.DataArray, np.array] = None,
    ) -> DataArray:
        """
        Calculate energy used and recuperated for a given vehicle per km driven.

        :param driving_mass: Mass of vehicle (kg)
        :param rr_coef: Rolling resistance coefficient (dimensionless, between 0.0 and 1.0)
        :param drag_coef: Aerodynamic drag coefficient (dimensionless, between 0.0 and 1.0)
        :param frontal_area: Frontal area of vehicle (m2)
        :param sizes: size classes of the vehicles
        :param electric_motor_power: Electric motor power (watts). Optional.
        :returns: net motive energy (in kJ/km)

        Power to overcome rolling resistance is calculated by:

        .. math::

            g v M C_{r}

        where :math:`g` is 9.81 (m/s2), :math:`v` is velocity (m/s), :math:`M` is mass (kg),
        and :math:`C_{r}` is the rolling resistance coefficient (dimensionless).

        Power to overcome air resistance is calculated by:

        .. math::

            \frac{1}{2} \rho_{air} v^{3} A C_{d}


        where :math:`\rho_{air}` is 1.225 (kg/m3), :math:`v` is velocity (m/s), :math:`A` is frontal area (m2), and :math:`C_{d}`
        is the aerodynamic drag coefficient (dimensionless).

        """

        _c = lambda x: x.values if isinstance(x, xr.DataArray) else x
        _o = lambda x: np.where((x == 0) | (x == np.nan), 1, x)

        # Calculate the energy used for each second of the drive cycle
        ones = np.ones_like(self.velocity)

        # Resistance from the tire rolling: rolling resistance coefficient * driving mass * 9.81
        rolling_resistance = _c((driving_mass * rr_coef * 9.81).T) * (self.velocity > 0)

        # Resistance from the drag: frontal area * drag coefficient * air density * 1/2 * velocity^2
        air_resistance = _c((frontal_area * drag_coef * self.rho_air / 2).T) * np.power(
            self.velocity, 2
        )

        # Resistance from road gradient: driving mass * 9.81 * sin(gradient)
        gradient_resistance = (
            _c((driving_mass * 9.81).T)
            * np.sin(np.nan_to_num(self.gradient)[:, None, None, None, :])
            * (self.velocity > 0)
        )

        # Inertia: driving mass * acceleration
        inertia = self.acceleration * _c(driving_mass).T

        total_resistance = (
            rolling_resistance + air_resistance + gradient_resistance + inertia
        )
        motive_energy_at_wheels = xr.where(total_resistance < 0, 0, total_resistance)
        motive_energy = np.zeros_like(motive_energy_at_wheels)

        # determining efficiencies
        engine_load = np.ones_like(motive_energy_at_wheels)

        if engine_efficiency is None:
            engine_efficiency = np.ones_like(motive_energy_at_wheels)

        if transmission_efficiency is None:
            transmission_efficiency = np.ones_like(motive_energy_at_wheels)

        engine_load_iterations = [0, engine_load.mean()]

        # we loop while the last three iterations are roughly equal
        # or while len(engine_load_iterations) < 10

        while len(engine_load_iterations) < 10:
            engine_efficiency = self.calculate_efficiency(
                engine_efficiency, engine_load, "engine"
            )

            transmission_efficiency = self.calculate_efficiency(
                transmission_efficiency, engine_load, "transmission"
            )

            recuperation_efficiency = xr.where(
                recuperation_efficiency == 0, 1, recuperation_efficiency
            )

            if fuel_cell_system_efficiency is None:
                fuel_cell_system_efficiency = np.array([1.0])

            fuel_cell_system_efficiency = xr.where(
                fuel_cell_system_efficiency == 0, 1, fuel_cell_system_efficiency
            )

            motive_energy = motive_energy_at_wheels / (
                _o(_c(engine_efficiency))
                * _o(_c(transmission_efficiency))
                * _o(_c(fuel_cell_system_efficiency)).T[None, ...]
            )

            engine_load = np.clip(
                (motive_energy / (_o(_c(engine_power)).T * 1000)) * self.velocity, 0, 1
            )

            # add a minimum 5% engine load when the vehicle is idling
            engine_load = np.where(self.velocity == 0, 0.05, engine_load)
            engine_load *= self.driving_time
            engine_load_iterations.append(engine_load.mean())

        negative_motive_energy = xr.where(total_resistance > 0, 0, total_resistance)
        recuperated_energy = (
            negative_motive_energy
            * _c(recuperation_efficiency).T[None, ...]
            * _c(battery_charge_eff).T[None, ...]
            * _c(battery_discharge_eff).T[None, ...]
            * (_c(electric_motor_power).T[None, ...] > 0)
        )

        if hvac_power is None:
            auxiliary_energy = self.aux_energy_per_km(
                aux_power,
                engine_efficiency,
                hvac_power,
                battery_cooling_unit,
                battery_heating_unit,
                heat_pump_cop_cooling,
                heat_pump_cop_heating,
                cooling_consumption,
                heating_consumption,
            )
            cooling_energy, heating_energy, battery_cooling, battery_heating = (
                np.zeros_like(auxiliary_energy),
                np.zeros_like(auxiliary_energy),
                np.zeros_like(auxiliary_energy),
                np.zeros_like(auxiliary_energy),
            )
        else:
            (
                auxiliary_energy,
                cooling_energy,
                heating_energy,
                battery_cooling,
                battery_heating,
            ) = self.aux_energy_per_km(
                aux_power,
                engine_efficiency,
                hvac_power,
                battery_cooling_unit,
                battery_heating_unit,
                heat_pump_cop_cooling,
                heat_pump_cop_heating,
                cooling_consumption,
                heating_consumption,
            )

        auxiliary_energy *= self.velocity > 0

        all_arrays = np.concatenate(
            [
                _(rolling_resistance),
                _(air_resistance),
                _(gradient_resistance),
                _(inertia),
                _(motive_energy_at_wheels),
                _(motive_energy),
                _(negative_motive_energy),
                _(recuperated_energy),
                _(auxiliary_energy),
                _(cooling_energy),
                _(heating_energy),
                _(battery_cooling),
                _(battery_heating),
                _(engine_load),
                _(transmission_efficiency),
                _(engine_efficiency),
                _(self.velocity * np.ones_like(motive_energy)),
            ],
            axis=-1,
        )

        all_arrays[..., :-4] /= 1000
        all_arrays[..., :-9] *= _(self.velocity)

        all_arrays[..., 5] = np.where(
            all_arrays[..., 5] > _(engine_power).T * 1,
            _(engine_power).T * 1,
            all_arrays[..., 5],
        )
        all_arrays[..., 7] = np.where(
            all_arrays[..., 7] < _(electric_motor_power).T * -1,
            _(electric_motor_power).T * -1,
            all_arrays[..., 7],
        )

        return convert_to_xr(all_arrays).fillna(0)
