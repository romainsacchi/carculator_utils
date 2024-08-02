"""
inventory.py contains InventoryCalculation which provides all methods to solve inventories.
"""

import csv
import itertools
import re
import warnings
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyprind
import xarray as xr
import yaml
from numpy import dtype, ndarray
from scipy import sparse

from . import DATA_DIR
from .background_systems import BackgroundSystemModel
from .export import ExportInventory

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

IAM_FILES_DIR = DATA_DIR / "IAM"


RANGE_PARAM = {
    "two-wheeler": "range",
    "car": "range",
    "bus": "daily distance",
    "truck": "target range",
}


def check_func_unit(func_unit):
    """Check if func_unit is a valid functional unit."""
    if func_unit not in ["vkm", "pkm", "tkm"]:
        raise ValueError(
            f"Functional unit must be one of "
            f"'vkm', 'pkm', 'tkm', "
            f"not {func_unit}"
        )
    return func_unit


def check_scenario(scenario):
    """Check if scenario is a valid scenario."""
    valid_scenarios = ["SSP2-NPi", "SSP2-PkBudg1150", "SSP2-PkBudg500", "static"]
    if scenario not in valid_scenarios:
        raise ValueError(
            f"Scenario must be one of " f"{valid_scenarios}, " f"not {scenario}"
        )
    return scenario


def get_noise_emission_flows() -> dict:
    """Get noise emission flows from the noise emission file."""

    return {
        (
            f"noise, octave {i}, day time, {comp}",
            (f"octave {i}", "day time", comp),
            "joule",
        ): f"noise, octave {i}, day time, {comp}"
        for i in range(1, 9)
        for comp in ["urban", "suburban", "rural"]
    }


def get_exhaust_emission_flows() -> dict:
    with open(
        DATA_DIR / "emission_factors" / "exhaust_and_noise_flows.yaml",
        "r",
        encoding="utf-8",
    ) as stream:
        flows = yaml.safe_load(stream)["exhaust"]

    d_comp = {
        "urban": "urban air close to ground",
        "suburban": "non-urban air or from high stacks",
        "rural": "low population density, long-term",
    }

    return {
        (v, ("air", d_comp[comp]), "kilogram"): f"{k} direct emissions, {comp}"
        for k, v in flows.items()
        for comp in ["urban", "suburban", "rural"]
    }


def get_dict_impact_categories(method, indicator) -> dict:
    """
    Load a dictionary with available impact assessment
    methods as keys, and assessment level and categories as values.

    :return: dictionary
    :rtype: dict
    """
    filename = "dict_impact_categories.csv"
    filepath = DATA_DIR / "lcia" / filename
    if not filepath.is_file():
        raise FileNotFoundError(
            "The dictionary of impact categories could not be found."
        )

    csv_dict = {}

    with open(filepath, encoding="utf-8") as f:
        input_dict = csv.reader(f, delimiter=";")
        for row in input_dict:
            if row[0] == method and row[3] == indicator:
                csv_dict[row[2]] = {
                    "method": row[1],
                    "category": row[2],
                    "type": row[3],
                    "abbreviation": row[4],
                    "unit": row[5],
                    "source": row[6],
                }

    return csv_dict


def get_dict_input() -> dict:
    """
    Load a dictionary with tuple ("name of activity", "location", "unit",
    "reference product") as key, row/column
    indices as values.

    :return: dictionary with `label:index` pairs.
    :rtype: dict

    """
    filename = f"dict_inputs_A_matrix.csv"
    filepath = DATA_DIR / "IAM" / filename
    if not filepath.is_file():
        raise FileNotFoundError("The dictionary of activity labels could not be found.")

    with open(filepath, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        raw = list(reader)
        for _r, r in enumerate(raw):
            if len(r) == 3:
                r[1] = eval(r[1])
            raw[_r] = tuple(r)

        return {j: i for i, j in enumerate(list(raw))}


def format_array(array):
    # Transpose the array as needed
    transposed_array = array.transpose(
        "value",
        "parameter",
        "size",
        "powertrain",
        "year",
    )

    # Determine the new shape for the reshaping operation
    new_shape = (
        array.sizes["value"],
        array.sizes["parameter"],
        -1,
        array.sizes["year"],
    )

    # Reshape the array while keeping it as an xarray DataArray
    reshaped_array = transposed_array.data.reshape(new_shape)

    combined_coords = [
        " - ".join(list(x))
        for x in list(
            itertools.product(
                array.coords["size"].values, array.coords["powertrain"].values
            )
        )
    ]

    # Convert the reshaped numpy array back to xarray DataArray
    reshaped_dataarray = xr.DataArray(
        reshaped_array,
        dims=["value", "parameter", "combined_dim", "year"],
        coords=[
            array.coords["value"].values,
            array.coords["parameter"].values,
            combined_coords,
            array.coords["year"].values,
        ],
    )

    return reshaped_dataarray


class Inventory:
    """
    Build and solve the inventory for results characterization and inventory export

    :ivar vm: object from the VehicleModel class
    :ivar background_configuration: dictionary that contains choices for background system
    :ivar scenario: IAM energy scenario to use (
        "SSP2-NPi": Nationally implemented policies, limits temperature increase by 2100 to 3.3 degrees Celsius,
        "SSP2-PkBudg1150": limits temperature increase by 2100 to 2 degrees Celsius,
        "SSP2-PkBudg500": limits temperature increase by 2100 to 1.5 degrees Celsius,
        "static": no forward-looking modification of the background inventories).
        "SSP2-NPi" selected by default.)

    """

    def __init__(
        self,
        vm,
        background_configuration: dict = None,
        scenario: str = "SSP2-NPi",
        method: str = "recipe",
        indicator: str = "midpoint",
        functional_unit: str = "vkm",
    ) -> None:
        self.vm = vm

        self.scope = {
            "size": vm.array.coords["size"].values.tolist(),
            "powertrain": vm.array.coords["powertrain"].values.tolist(),
            "year": vm.array.coords["year"].values.tolist(),
        }
        self.scenario = check_scenario(scenario)
        self.func_unit = check_func_unit(functional_unit)

        self.method = method
        self.indicator = indicator if method == "recipe" else "midpoint"

        self.array = format_array(vm.array)
        self.iterations = len(vm.array.value.values)

        self.number_of_vehicles = (self.vm["TtW energy"] > 0).sum().values

        self.background_configuration = {}
        self.background_configuration.update(background_configuration or {})

        self.inputs = get_dict_input()

        self.bs = BackgroundSystemModel()
        self.add_additional_activities()
        self.rev_inputs = {v: k for k, v in self.inputs.items()}

        with open(
            DATA_DIR / "electricity" / "elec_tech_map.yaml", "r", encoding="utf-8"
        ) as stream:
            self.elec_map = yaml.safe_load(stream)
            self.elec_map = {k: tuple(v) for k, v in self.elec_map.items()}

        self.electricity_technologies = list(self.elec_map.keys())

        self.A = self.get_A_matrix()
        # Create electricity and fuel market datasets
        self.mix = self.define_electricity_mix_for_fuel_prep()
        self.create_electricity_mix_for_fuel_prep()
        self.rev_inputs = {v: k for k, v in self.inputs.items()}
        self.create_fuel_markets()

        self.exhaust_emissions = get_exhaust_emission_flows()
        self.noise_emissions = get_noise_emission_flows()

        self.list_cat, self.split_indices = self.get_split_indices()

        self.impact_categories = get_dict_impact_categories(
            method=self.method, indicator=self.indicator
        )

        # Create the B matrix
        self.B = self.get_B_matrix()
        self.rev_inputs = {v: k for k, v in self.inputs.items()}

        self.fill_in_A_matrix()
        self.remove_non_compliant_vehicles()

    def get_results_table(self, sensitivity: bool = False) -> xr.DataArray:
        """
        Format a xarray.DataArray array to receive the results.

        :param sensitivity: if True, the results table will
        be formatted to receive sensitivity analysis results
        :return: xarrray.DataArray
        """

        params = [a for a in self.array.value.values]
        response = xr.DataArray(
            np.zeros(
                (
                    len(self.impact_categories),
                    len(self.scope["size"]),
                    len(self.scope["powertrain"]),
                    len(self.scope["year"]),
                    len(self.list_cat),
                    self.iterations,
                )
            ),
            coords=[
                list(self.impact_categories.keys()),
                self.scope["size"],
                self.scope["powertrain"],
                self.scope["year"],
                self.list_cat,
                np.arange(0, self.iterations) if not sensitivity else params,
            ],
            dims=[
                "impact_category",
                "size",
                "powertrain",
                "year",
                "impact",
                "value",
            ],
        )

        if sensitivity:
            # remove the `impact` dimension
            response = response.sum(dim="impact")

        return response

    def get_split_indices(self):
        """
        Return list of indices to split the results into categories.

        :return: list of indices
        :rtype: list
        """
        # read `impact_source_categories.yaml` file
        with open(
            DATA_DIR / "lcia" / "impact_source_categories.yaml", "r", encoding="utf-8"
        ) as stream:
            source_cats = yaml.safe_load(stream)

        idx_cats = defaultdict(list)

        for cat, name in source_cats.items():
            for n in name:
                idx = self.find_input_indices((n,))
                if idx and idx not in idx_cats[cat]:
                    idx_cats[cat].extend(self.find_input_indices((n,)))
            # remove duplicates
            idx_cats[cat] = list(set(idx_cats[cat]))

        # add flows corresponding to `exhaust - direct`
        idx_cats["direct - exhaust"] = [
            self.inputs[("Carbon dioxide, fossil", ("air",), "kilogram")],
            self.inputs[("Carbon dioxide, non-fossil", ("air",), "kilogram")],
        ]
        idx_cats["direct - exhaust"].append(
            self.inputs[("Sulfur dioxide", ("air",), "kilogram")]
        )
        idx_cats["direct - exhaust"].extend(
            [self.inputs[i] for i in self.exhaust_emissions]
        )
        idx_cats["direct - exhaust"].extend(
            [self.inputs[i] for i in self.noise_emissions]
        )

        idx_cats["direct - non-exhaust"].append(
            self.inputs[("Methane, fossil", ("air",), "kilogram")],
        )

        # idx for an input that has no burden
        # oxygen in this case
        extra_idx = [j for i, j in self.inputs.items() if i[0].lower() == "oxygen"][0]

        list_ind = [val for val in idx_cats.values()]
        maxLen = max(map(len, list_ind))
        for row in list_ind:
            while len(row) < maxLen:
                row.append(extra_idx)

        return list(idx_cats.keys()), list_ind

    def get_load_factor(self):
        # If the FU is in passenger-km, we normalize the results by
        # the number of passengers
        if self.func_unit == "vkm":
            load_factor = 1
        elif self.func_unit == "pkm":
            load_factor = self.array.sel(parameter="average passengers")
            load_factor = np.resize(
                load_factor.values,
                (
                    1,
                    len(self.scope["size"]),
                    len(self.scope["powertrain"]),
                    len(self.scope["year"]),
                    1,
                    1,
                ),
            )
        else:
            # ton kilometers
            load_factor = np.resize(
                self.array[self.array_inputs["cargo mass"]].values / 1000,
                (
                    1,
                    len(self.scope["size"]),
                    len(self.scope["powertrain"]),
                    len(self.scope["year"]),
                    1,
                    1,
                ),
            )

        return load_factor

    def calculate_impacts(self, sensitivity=False):
        if self.scenario != "static":
            B = self.B.interp(
                year=self.scope["year"], kwargs={"fill_value": "extrapolate"}
            ).values
        else:
            B = self.B.values

        # Prepare an array to store the results
        results = self.get_results_table(sensitivity=sensitivity)

        new_arr = np.zeros((self.A.shape[1], self.B.shape[1], self.A.shape[-1]))

        f_vector = np.zeros((np.shape(self.A)[1]))

        # Collect indices of activities contributing to the first level
        idx_car_trspt = self.find_input_indices(
            (f"transport, {self.vm.vehicle_type}, ",)
        )
        idx_cars = self.find_input_indices((f"{self.vm.vehicle_type.capitalize()}, ",))

        idx_others = [
            i
            for i in self.inputs.values()
            if not any(i in x for x in [idx_car_trspt, idx_cars])
        ]

        arr = (
            self.A[
                np.ix_(
                    np.arange(self.iterations),
                    idx_others,
                    np.array(idx_cars + idx_car_trspt),
                )
            ]
            .sum(axis=0)
            .sum(axis=1)
        )

        nonzero_idx = np.argwhere(arr)

        # use pyprind to display a progress bar
        bar = pyprind.ProgBar(len(nonzero_idx), stream=1, title="Calculating impacts")

        for a in nonzero_idx:
            bar.update()

            if isinstance(self.rev_inputs[a[0]][1], tuple):
                # it's a biosphere flow, hence no need to calculate LCA
                new_arr[a[0], :, a[1]] = B[a[1], :, a[0]]

            else:
                f_vector[:] = 0
                f_vector[a[0]] = 1
                X = sparse.linalg.spsolve(
                    sparse.csr_matrix(self.A[0, ..., a[1]]), f_vector.T
                )
                _X = (X * B[a[1]]).sum(axis=-1).T
                new_arr[a[0], :, a[1]] = _X

        new_arr = new_arr.transpose(1, 0, 2)

        arr = (
            self.A[:, :, idx_car_trspt].reshape(
                self.iterations,
                -1,
                len(self.scope["size"]),
                len(self.scope["powertrain"]),
                len(self.scope["year"]),
            )
            * new_arr[:, None, :, None, None, :]
            * -1
        )

        arr += (
            self.A[:, :, idx_cars].reshape(
                self.iterations,
                -1,
                len(self.scope["size"]),
                len(self.scope["powertrain"]),
                len(self.scope["year"]),
            )
            * new_arr[:, None, :, None, None, :]
            * self.A[:, idx_cars, idx_car_trspt].reshape(
                self.iterations,
                -1,
                len(self.scope["size"]),
                len(self.scope["powertrain"]),
                len(self.scope["year"]),
            )
        )

        arr = arr[:, :, self.split_indices].sum(axis=3)

        # fetch indices not contained in self.split_indices
        # to see if there are other flows unaccounted for
        idx = [
            i
            for i in range(self.B.shape[-1])
            if i not in list(itertools.chain.from_iterable(self.split_indices))
        ]
        # check if any of the first items of nonzero_idx
        # are in idx
        for i in nonzero_idx:
            if i[0] in idx:
                print(f"The flow {self.rev_inputs[i[0]][0]} is not accounted for.")

        # reshape the array to match the dimensions of the results table
        arr = arr.transpose(0, 3, 4, 5, 2, 1)

        if sensitivity:
            results[...] = arr.sum(axis=-2)
            results /= results.sel(value="reference")
        else:
            results[...] = arr

        load_factor = self.get_load_factor()

        # check that load_factor has the same number of dimensions
        # otherwise, resize it
        if isinstance(load_factor, np.ndarray):
            if load_factor.ndim > results.ndim:
                load_factor = np.resize(
                    load_factor,
                    (
                        1,
                        len(self.scope["size"]),
                        len(self.scope["powertrain"]),
                        len(self.scope["year"]),
                        1,
                    ),
                )

        return results / load_factor

    def add_additional_activities(self):
        # Add as many rows and columns as cars to consider
        # Also add additional columns and rows for electricity markets
        # for fuel preparation and energy battery production

        maximum = max(self.inputs.values())

        for fuel in ["petrol", "diesel", "hydrogen", "methane"]:
            maximum += 1
            self.inputs[
                (
                    f"fuel supply for {fuel} vehicles",
                    self.vm.country,
                    "kilogram",
                    "fuel",
                )
            ] = maximum

        for electricity_source in [
            (f"electricity supply for electric vehicles", self.vm.country),
            (f"electricity supply for fuel preparation", self.vm.country),
        ]:
            maximum += 1
            self.inputs[
                (
                    electricity_source[0],
                    electricity_source[1],
                    "kilowatt hour",
                    "electricity, low voltage",
                )
            ] = maximum

        with open(DATA_DIR / "emission_factors" / "euro_classes.yaml", "r") as stream:
            euro_classes = yaml.safe_load(stream)[self.vm.vehicle_type]

        list_years = np.clip(
            self.scope["year"],
            min(euro_classes.keys()),
            max(euro_classes.keys()),
        )

        list_euro_classes = [euro_classes[y] for y in list(list_years)]

        for size in self.scope["size"]:
            for powertrain in self.scope["powertrain"]:
                for euro_class, year in zip(list_euro_classes, self.scope["year"]):
                    if self.func_unit == "vkm":
                        unit = "kilometer"
                    elif self.func_unit == "pkm":
                        unit = "passenger kilometer"
                    else:
                        unit = "ton kilometer"

                    if powertrain in ["BEV", "BEV-opp", "BEV-depot", "BEV-motion"]:
                        chemistry = self.vm.energy_storage["electric"][
                            (powertrain, size, year)
                        ]
                        name = f"transport, {self.vm.vehicle_type}, {powertrain}, {chemistry} battery, {size}"
                        ref = f"transport, {self.vm.vehicle_type}"

                    elif powertrain in ["FCEV", "Human"]:
                        name = (
                            f"transport, {self.vm.vehicle_type}, {powertrain}, {size}"
                        )
                        ref = f"transport, {self.vm.vehicle_type}"

                    else:
                        name = (
                            f"transport, {self.vm.vehicle_type}, {powertrain}, {size}"
                        )
                        ref = f"transport, {self.vm.vehicle_type}"

                    # add transport activity
                    key = (name, self.vm.country, unit, ref)
                    if key not in self.inputs:
                        maximum += 1
                        self.inputs[(name, self.vm.country, unit, ref)] = maximum

                    # add vehicle
                    key = (
                        name.replace(
                            f"transport, {self.vm.vehicle_type}",
                            self.vm.vehicle_type.capitalize(),
                        ),
                        self.vm.country,
                        "unit",
                        ref.replace(
                            f"transport, {self.vm.vehicle_type}",
                            self.vm.vehicle_type.capitalize(),
                        ),
                    )

                    if key not in self.inputs:
                        maximum += 1
                        self.inputs[key] = maximum

    def get_A_matrix(self):
        """
        Load the A matrix. The matrix contains exchanges of products (rows)
        between activities (columns).

        :return: A matrix with three dimensions of shape (number of values,
        number of products, number of activities).
        :rtype: numpy.ndarray

        """

        filename = "A_matrix.npz"
        filepath = DATA_DIR / "IAM" / filename
        if not filepath.is_file():
            raise FileNotFoundError("The IAM files could not be found.")

        # load matrix A
        initial_A = sparse.load_npz(filepath).toarray()

        new_A = np.identity(len(self.inputs))
        new_A[0 : np.shape(initial_A)[0], 0 : np.shape(initial_A)[0]] = initial_A

        # Resize the matrix to fit the number of `value` in `self.array`
        new_A = np.resize(
            new_A,
            (
                self.iterations,
                len(self.inputs),
                len(self.inputs),
            ),
        )

        # add a `year`dimension, with length equal to the number of years
        # in the scope
        new_A = np.repeat(new_A[:, :, :, None], len(self.scope["year"]), axis=-1)

        return new_A

    def get_B_matrix(self) -> xr.DataArray:
        """
        Load the B matrix. The B matrix contains impact assessment
        figures for a give impact assessment method,
        per unit of activity. Its length column-wise equals
        the length of the A matrix row-wise.
        Its length row-wise equals the number of
        impact assessment methods.

        :return: an array with impact values per unit
        of activity for each method.
        :rtype: numpy.ndarray

        """

        filepaths = [
            str(fp)
            for fp in list(Path(IAM_FILES_DIR).glob("*.npz"))
            if all(x in str(fp) for x in [self.method, self.indicator, self.scenario])
        ]

        if self.scenario != "static":
            filepaths = sorted(filepaths, key=lambda x: int(x[-8:-4]))

        B = np.zeros((len(filepaths), len(self.impact_categories), len(self.inputs)))

        for f, filepath in enumerate(filepaths):
            initial_B = sparse.load_npz(filepath).toarray()

            new_B = np.zeros(
                (
                    initial_B.shape[0],
                    len(self.inputs),
                )
            )

            new_B[0 : initial_B.shape[0], 0 : initial_B.shape[1]] = initial_B
            B[f, :, :] = new_B

        return xr.DataArray(
            B,
            coords=[
                (
                    [2005, 2010, 2020, 2030, 2040, 2050]
                    if self.scenario != "static"
                    else [2020]
                ),
                np.asarray(list(self.impact_categories.keys()), dtype="object"),
                np.asarray(list(self.inputs.keys()), dtype="object"),
            ],
            dims=["year", "category", "activity"],
        )

    def get_index_of_flows(self, items_to_look_for, search_by="name"):
        """
        Return list of row/column indices of self.A of labels that contain the string defined in `items_to_look_for`.

        :param items_to_look_for: string
        :param search_by: "name" or "compartment" (for elementary flows)
        :return: list of row/column indices
        :rtype: list
        """
        if search_by == "name":
            return [
                int(self.inputs[c])
                for c in self.inputs
                if all(ele in c[0].lower() for ele in items_to_look_for)
            ]
        if search_by == "compartment":
            return [
                int(self.inputs[c])
                for c in self.inputs
                if all(ele in c[1] for ele in items_to_look_for)
            ]

    def define_electricity_mix_for_fuel_prep(self) -> np.ndarray:
        """
        This function defines a fuel mix based either on user-defined mix,
        or on default mixes for a given country.
        The mix is calculated as the average mix, weighted by the
        distribution of annually driven kilometers.
        :return:
        """
        try:
            losses_to_low = float(self.bs.losses[self.vm.country]["LV"])
        except KeyError:
            # If losses for the country are not found, assume EU average
            losses_to_low = float(self.bs.losses["RER"]["LV"])

        if "custom electricity mix" in self.background_configuration:
            # If a special electricity mix is specified, we use it
            mix = self.background_configuration["custom electricity mix"]

            if np.shape(mix)[0] != len(self.scope["year"]):
                raise ValueError(
                    "The number of electricity mixes ({}) must match with the "
                    "number of years ({}).".format(
                        np.shape(mix)[0], len(self.scope["year"])
                    )
                )

            if not np.allclose(np.sum(mix, 1), np.ones(len(self.scope["year"]))):
                print(
                    "The sum of the electricity mix share does "
                    "not equal to 1 for each year."
                )

        else:
            use_year = (
                self.array.sel(parameter="lifetime kilometers")
                / self.array.sel(parameter="kilometers per year")
            ).mean(dim=["combined_dim", "value"])

            # create an array that contain integers starting from self.scope["year"]
            # to self.scope["year"] + use_year, e.g., 2020, 2021, 2022, ..., 2035

            use_year = np.array(
                [np.arange(y, y + u) for y, u in zip(self.scope["year"], use_year)]
            )

            if self.vm.country not in self.bs.electricity_mix.country.values:
                print(
                    f"The electricity mix for {self.vm.country} could not be found."
                    "Average European electricity mix is used instead."
                )
                country = "RER"
            else:
                country = self.vm.country

            mix = [
                (
                    self.bs.electricity_mix.sel(
                        country=country,
                        variable=self.electricity_technologies,
                    )
                    .interp(
                        year=use_year[y],
                        kwargs={"fill_value": "extrapolate"},
                    )
                    .mean(axis=0)
                    .values
                    if use_year[y][-1] <= 2050
                    else self.bs.electricity_mix.sel(
                        country=country,
                        variable=self.electricity_technologies,
                    )
                    .interp(
                        year=np.arange(year, 2051), kwargs={"fill_value": "extrapolate"}
                    )
                    .mean(axis=0)
                    .values
                )
                for y, year in enumerate(self.scope["year"])
            ]

        return np.clip(mix, 0, 1) / np.clip(mix, 0, 1).sum(axis=1)[:, None]

    def define_renewable_rate_in_mix(self) -> ndarray[Any, dtype[Any]]:
        """
        This function returns the renewable rate in the electricity mix
        for each year.
        """

        sum_non_hydro_renew = [
            np.sum([mix[i] for i in [3, 4, 5, 8, 9, 10, 11, 14, 18, 19]])
            for mix in self.mix
        ]

        sum_hydro = [np.sum([mix[i] for i in [0, 15]]) for mix in self.mix]

        sum_nuclear = [
            np.sum(
                [
                    mix[i]
                    for i in [
                        1,
                    ]
                ]
            )
            for mix in self.mix
        ]

        rates = np.vstack(
            (
                sum_non_hydro_renew,
                sum_hydro,
                sum_nuclear,
            )
        )

        return rates

    # @lru_cache
    def find_input_indices(self, contains: [tuple, str], excludes: tuple = ()) -> list:
        """
        This function finds the indices of the inputs in the A matrix
        that contain the strings in the contains list, and do not
        contain the strings in the excludes list.
        :param contains: list of strings
        :param excludes: list of strings
        :return: list of indices
        """
        indices = []

        if not isinstance(contains, tuple):
            contains = tuple(contains)

        if not isinstance(excludes, tuple):
            excludes = tuple(excludes)

        for i, input in enumerate(self.inputs):
            if all([c in input[0] for c in contains]) and not any(
                [e in input[0] for e in excludes]
            ):
                indices.append(i)

        return indices

    def add_electricity_infrastructure(self, dataset, losses):
        # Add transmission network for high and medium voltage
        for input in [
            (
                ("transmission network construction, electricity, high voltage",),
                dataset,
                6.58e-9 * -1 * losses,
            ),
            (
                ("transmission network construction, electricity, medium voltage",),
                dataset,
                1.86e-8 * -1 * losses,
            ),
            (
                ("distribution network construction, electricity, low voltage",),
                dataset,
                8.74e-8 * -1 * losses,
            ),
            (
                ("market for sulfur hexafluoride, liquid",),
                dataset,
                (5.4e-8 + 2.99e-9) * -1 * losses,
            ),
            (("Sulfur hexafluoride",), dataset, (5.4e-8 + 2.99e-9) * -1 * losses),
        ]:
            self.A[
                np.ix_(
                    np.arange(self.iterations),
                    self.find_input_indices(
                        input[0],
                    )[:1],
                    self.find_input_indices((input[1],)),
                )
            ] = input[2]

    def create_electricity_mix_for_fuel_prep(self):
        """
        This function fills the electricity market that
        supplies battery charging operations
        and hydrogen production through electrolysis.
        """

        try:
            losses_to_low = float(self.bs.losses[self.vm.country]["LV"])
        except KeyError:
            # If losses for the country are not found, assume EU average
            losses_to_low = float(self.bs.losses["RER"]["LV"])

        # Fill the electricity markets for battery charging and hydrogen production
        # Add electricity technology shares
        self.A[
            np.ix_(
                np.arange(self.iterations),
                [self.inputs[self.elec_map[t]] for t in self.electricity_technologies],
                self.find_input_indices(("electricity supply for fuel preparation",)),
            )
        ] = (self.mix * -1 * losses_to_low).T[None, :, None, :]

        self.add_electricity_infrastructure(
            "electricity supply for fuel preparation", losses_to_low
        )

    def get_sulfur_content(self, location, fuel):
        """
        Return the sulfur content in the fuel.
        If a region is passed, the average sulfur content over
        the countries the region contains is returned.

        :param year:
        :param location: str. A country or region ISO code
        :param fuel: str. "diesel" or "petrol"
        :return: float. Sulfur content in ppm.
        """

        if fuel not in self.bs.sulfur.fuel.values:
            return 0

        if location in self.bs.sulfur.country.values:
            sulfur_concentration = (
                self.bs.sulfur.sel(country=location, year=self.scope["year"], fuel=fuel)
                .sum()
                .values
            )
        else:
            # If the geography is not found,
            # we use the European average

            print(
                f"The sulfur content for {fuel} fuel in {location} "
                f"could not be found."
                "European average sulfur content is used instead."
            )

            sulfur_concentration = (
                self.bs.sulfur.sel(country="RER", year=self.scope["year"], fuel=fuel)
                .sum()
                .values
            )

        return sulfur_concentration

    def create_fuel_markets(self):
        """
        This function creates markets for fuel, considering a given blend,
        a given fuel type and a given year.
        It also adds separate electricity input in case hydrogen
        from electrolysis is needed somewhere in the fuel supply chain.
        :return:
        """

        d_dataset_name = {
            "petrol": "fuel supply for petrol vehicles",
            "diesel": "fuel supply for diesel vehicles",
            "methane": "fuel supply for methane vehicles",
            "hydrogen": "fuel supply for hydrogen vehicles",
        }

        # electricity dataset
        # for y, year in enumerate(self.scope["year"]):
        self.A[
            :,
            self.find_input_indices(("electricity supply for fuel preparation",)),
            self.find_input_indices((f"electricity supply for electric vehicles",)),
        ] = -1

        # fuel datasets
        for fuel_type in self.vm.fuel_blend:
            self.find_input_requirement(
                value_in="kilowatt hour",
                value_out=self.vm.fuel_blend[fuel_type]["primary"]["name"][0],
                find_input_by="unit",
                replace_by=self.find_input_indices(
                    ("electricity supply for fuel preparation",)
                ),
            )
            self.find_input_requirement(
                value_in="kilowatt hour",
                value_out=self.vm.fuel_blend[fuel_type]["secondary"]["name"][0],
                find_input_by="unit",
                replace_by=self.find_input_indices(
                    ("electricity supply for fuel preparation",)
                ),
            )

            for y, year in enumerate(self.scope["year"]):
                primary_share = self.vm.fuel_blend[fuel_type]["primary"]["share"][y]
                secondary_share = self.vm.fuel_blend[fuel_type]["secondary"]["share"][y]
                fuel_market_index = self.find_input_indices(
                    (d_dataset_name[fuel_type],)
                )

                try:
                    primary_fuel_activity_index = self.inputs[
                        self.vm.fuel_blend[fuel_type]["primary"]["name"]
                    ]
                    secondary_fuel_activity_index = self.inputs[
                        self.vm.fuel_blend[fuel_type]["secondary"]["name"]
                    ]
                except KeyError:
                    raise KeyError(
                        "One of the primary or secondary fuels specified in "
                        "the fuel blend for {} is not valid.".format(fuel_type)
                    )

                self.A[:, primary_fuel_activity_index, fuel_market_index, y] = (
                    -1 * primary_share
                )
                self.A[:, secondary_fuel_activity_index, fuel_market_index, y] = (
                    -1 * secondary_share
                )

    def find_input_requirement(
        self,
        value_in,
        value_out,
        find_input_by="name",
        zero_out_input=False,
        filter_activities=None,
        replace_by=None,
    ):
        """
        Finds the exchange inputs to a specified functional unit
        :param zero_out_input:
        :param find_input_by: can be 'name' or 'unit'
        :param value_in: value to look for
        :param value_out: functional unit output
        :return: indices of all inputs to FU, indices of inputs of interest
        :rtype: tuple
        """

        if isinstance(value_out, str):
            value_out = (value_out,)

        index_output = self.find_input_indices(value_out)

        f_vector = np.zeros((np.shape(self.A)[1]))
        f_vector[index_output] = 1

        X = sparse.linalg.spsolve(sparse.csr_matrix(self.A[0, ..., 0]), f_vector.T)

        ind_inputs = np.nonzero(X)[0]

        if find_input_by == "name":
            ins = [
                i
                for i in ind_inputs
                if value_in.lower() in self.rev_inputs[i][0].lower()
            ]

        elif find_input_by == "unit":
            ins = [
                i
                for i in ind_inputs
                if value_in.lower() in self.rev_inputs[i][2].lower()
            ]
        else:
            raise ValueError("find_input_by must be 'name' or 'unit'")

        outs = [i for i in ind_inputs if i not in ins]

        if filter_activities:
            outs = [
                i
                for e in filter_activities
                for i in outs
                if e.lower() in self.rev_inputs[i][0].lower()
            ]

        ins = [
            i
            for i in ins
            if self.A[np.ix_(np.arange(0, self.A.shape[0]), [i], outs)].sum() != 0
        ]

        # if replace_by, replace the input by the new one
        if replace_by:
            for i in ins:
                amount = self.A[np.ix_(np.arange(0, self.A.shape[0]), [i], outs)]
                self.A[np.ix_(np.arange(0, self.A.shape[0]), [i], outs)] = 0
                self.A[np.ix_(np.arange(0, self.A.shape[0]), replace_by, outs)] = amount

            return

        sum_supplied = X[ins].sum()

        if zero_out_input:
            # zero out initial inputs
            self.A[np.ix_(np.arange(0, self.A.shape[0]), ins, outs)] = 0

        return sum_supplied

    def get_fuel_blend_carbon_intensity(
        self, fuel_type: str
    ) -> [np.ndarray, np.ndarray]:
        """
        Returns the carbon intensity of a fuel blend.
        :param fuel_type: fuel type
        :return: carbon intensity of fuel blend fossil, and biogenic
        """
        primary_share = self.vm.fuel_blend[fuel_type]["primary"]["share"]
        secondary_share = self.vm.fuel_blend[fuel_type]["secondary"]["share"]

        primary_CO2 = self.vm.fuel_blend[fuel_type]["primary"]["CO2"]
        secondary_CO2 = self.vm.fuel_blend[fuel_type]["secondary"]["CO2"]

        primary_biogenic_share = self.vm.fuel_blend[fuel_type]["primary"][
            "biogenic share"
        ]
        secondary_biogenic_share = self.vm.fuel_blend[fuel_type]["secondary"][
            "biogenic share"
        ]

        return (
            primary_share * primary_CO2 * (1 - primary_biogenic_share)
            + secondary_share * secondary_CO2 * (1 - secondary_biogenic_share),
            primary_share * primary_CO2 * primary_biogenic_share
            + secondary_share * secondary_CO2 * secondary_biogenic_share,
        )

    def fill_in_A_matrix(self):
        """
        Fill-in the A matrix. Does not return anything. Modifies in place.
        Shape of the A matrix (values, products, activities).

        :param array: :attr:`array` from :class:`CarModel` class
        """

        pass

    def add_fuel_cell_stack(self):
        self.A[
            :,
            self.find_input_indices(("ancillary BoP components for fuel cell system",)),
            self.find_input_indices((f"{self.vm.vehicle_type.capitalize()}, ",)),
        ] = (
            self.array.sel(parameter="fuel cell ancillary BoP mass") * -1
        )

        self.A[
            :,
            self.find_input_indices(("essential BoP components for fuel cell system",)),
            self.find_input_indices((f"{self.vm.vehicle_type.capitalize()}, ",)),
        ] = (
            self.array.sel(parameter="fuel cell essential BoP mass") * -1
        )

        # note: `Stack`refers to the power of the stack, not mass
        self.A[
            :,
            self.find_input_indices(
                contains=("fuel cell stack production, 1 kW",), excludes=("PEM",)
            ),
            self.find_input_indices((f"{self.vm.vehicle_type.capitalize()}, ",)),
        ] = (
            self.array.sel(parameter="fuel cell power")
            * (1 + self.array.sel(parameter="fuel cell lifetime replacements"))
            * -1
        )

    def add_hydrogen_tank(self):
        hydro_tank_type = self.vm.energy_storage.get(
            "hydrogen", {"tank type": "carbon fiber"}
        )["tank type"]

        dict_tank_map = {
            "carbon fiber": "fuel tank production, compressed hydrogen gas, 700bar, with carbon fiber",
            "hdpe": "fuel tank production, compressed hydrogen gas, 700bar",
            "aluminium": "fuel tank production, compressed hydrogen gas, 700bar, with aluminium liner",
        }

        self.A[
            :,
            self.find_input_indices((dict_tank_map[hydro_tank_type],)),
            self.find_input_indices((f"{self.vm.vehicle_type.capitalize()}, ",)),
        ] = (
            self.array.sel(parameter="fuel tank mass")
            * (self.array.sel(parameter="fuel cell power") > 0)
            * -1
        )

    def add_battery(self):
        # Start of printout
        print(
            "****************** IMPORTANT BACKGROUND PARAMETERS ******************",
            end="\n * ",
        )

        # Energy storage
        print(f"The functional unit is: {self.func_unit}.", end="\n * ")
        print(f"The background prospective scenario is: {self.scenario}.", end="\n * ")
        print(f"The country of use is: {self.vm.country}.", end="\n * ")

        battery_tech = list(set(list(self.vm.energy_storage["electric"].values())))
        if len(battery_tech) == 0:
            battery_tech = ["NMC-622"]

        battery_origin = self.vm.energy_storage.get("origin", "CN")

        print(
            "Power and energy batteries produced "
            f"in {battery_origin} using {battery_tech} chemistry/ies",
            end="\n",
        )

        #  battery BoP for all vehicles
        self.A[
            :,
            self.find_input_indices(("battery BoP",)),
            self.find_input_indices((f"{self.vm.vehicle_type.capitalize()}, ",)),
        ] = (
            self.array.sel(parameter="battery BoP mass")
            * (1 + self.array.sel(parameter="battery lifetime replacements"))
            * -1
        )

        # Use the NMC inventory for all non-electric vehicles
        # because they need one in the future as they become hybrid.
        for pwt in [
            p
            for p in self.scope["powertrain"]
            if p
            in [
                "ICEV-p",
                "ICEV-d",
                "ICEV-g",
            ]
        ]:
            for size in self.scope["size"]:
                for year in self.scope["year"]:
                    if (pwt, size, year) not in self.vm.energy_storage["electric"]:
                        self.vm.energy_storage["electric"][
                            (pwt, size, year)
                        ] = "NMC-622"

        for key, val in self.vm.energy_storage["electric"].items():
            pwt, size, year = key
            self.A[
                :,
                self.find_input_indices((f"battery cell, {val}",)),
                self.find_input_indices(
                    (
                        f"{self.vm.vehicle_type.capitalize()}, ",
                        pwt,
                        size,
                    )
                ),
                self.scope["year"].index(year),
            ] = (
                self.array.sel(
                    parameter="battery cell mass",
                    combined_dim=[
                        d
                        for d in self.array.coords["combined_dim"].values
                        if all(x in d for x in [pwt, size])
                    ],
                    year=year,
                )
                * (
                    1
                    + self.array.sel(
                        parameter="battery lifetime replacements",
                        combined_dim=[
                            d
                            for d in self.array.coords["combined_dim"].values
                            if all(x in d for x in [pwt, size])
                        ],
                        year=year,
                    )
                )
                * -1
            )

        # Battery EoL
        self.A[
            :,
            self.find_input_indices(("market for used Li-ion battery",)),
            self.find_input_indices((f"{self.vm.vehicle_type.capitalize()}, ",)),
        ] = self.array.sel(parameter=["battery cell mass", "battery BoP mass"]).sum(
            dim="parameter"
        ) * (
            1 + self.array.sel(parameter="battery lifetime replacements")
        )

    def add_cng_tank(self):
        self.A[
            :,
            self.find_input_indices(
                contains=("fuel tank production, compressed natural gas, 200 bar",)
            ),
            self.find_input_indices(
                contains=(f"{self.vm.vehicle_type.capitalize()}, ", "ICEV-g")
            ),
        ] = (
            self.array.sel(
                parameter="fuel tank mass",
                combined_dim=[
                    d for d in self.array.coords["combined_dim"].values if "ICEV-g" in d
                ],
            )
            * -1
        )

    def add_vehicle_to_transport_dataset(self):
        self.A[
            :,
            self.find_input_indices((f"{self.vm.vehicle_type.capitalize()}, ",)),
            self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",)),
        ] = -1 / self.array.sel(parameter="lifetime kilometers")

    def display_renewable_rate_in_mix(self):
        sum_renew = self.define_renewable_rate_in_mix()

        use_year = (
            self.array.sel(parameter="lifetime kilometers")
            / self.array.sel(parameter="kilometers per year")
        ).mean(dim=["combined_dim", "value"])

        # create an array that contain integers starting from self.scope["year"]
        # to self.scope["year"] + use_year, e.g., 2020, 2021, 2022, ..., 2035

        use_year = np.array(
            [np.arange(y, y + u) for y, u in zip(self.scope["year"], use_year)]
        )

        for y, year in enumerate(self.scope["year"]):
            print(
                f"\t * between {int(np.min(use_year[y]))} and {int(np.max(use_year[y]))}, "
                f"% of non-hydro renew.: {int(sum_renew[0][y] * 100)}, "
                f"hydro: {int(sum_renew[1][y] * 100)}, "
                f"nuclear: {int(sum_renew[2][y] * 100)}.",
            )

    def add_electricity_to_electric_vehicles(self) -> None:
        electric_powertrains = [
            "BEV",
            "BEV-opp",
            "BEV-motion",
            "BEV-depot",
            "PHEV-p",
            "PHEV-d",
        ]

        if any(True for x in electric_powertrains if x in self.scope["powertrain"]):
            self.A[
                np.ix_(
                    np.arange(self.iterations),
                    self.find_input_indices(
                        (f"electricity supply for electric vehicles",)
                    ),
                    self.find_input_indices(
                        contains=(f"transport, {self.vm.vehicle_type}, ",),
                    ),
                )
            ] = (
                self.array.sel(parameter=["electricity consumption"]) * -1
            )

    def add_hydrogen_to_fuel_cell_vehicles(self) -> None:
        if "FCEV" in self.scope["powertrain"]:
            print(
                "{} is completed by {}.".format(
                    self.vm.fuel_blend["hydrogen"]["primary"]["type"],
                    self.vm.fuel_blend["hydrogen"]["secondary"]["type"],
                ),
                end="\n \t * ",
            )

            for y, year in enumerate(self.scope["year"]):
                if y + 1 == len(self.scope["year"]):
                    end_str = "\n * "
                else:
                    end_str = "\n \t * "

                print(
                    f"in {year} _________________________________________ "
                    f"{np.round(self.vm.fuel_blend['hydrogen']['secondary']['share'][y]* 100)}%",
                    end=end_str,
                )

            # Fuel supply
            self.A[
                :,
                self.find_input_indices(("fuel supply for hydrogen vehicles",)),
                self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",)),
            ] = (
                self.array.sel(parameter="fuel consumption")
                * self.array.sel(parameter="fuel density per kg")
                * (self.array.sel(parameter="fuel cell power") > 0)
                * -1
            )

    def display_fuel_blend(self, fuel) -> None:
        print(
            "{} is completed by {}.".format(
                self.vm.fuel_blend[fuel]["primary"]["type"],
                self.vm.fuel_blend[fuel]["secondary"]["type"],
            ),
            end="\n \t * ",
        )

        for y, year in enumerate(self.scope["year"]):
            if y + 1 == len(self.scope["year"]):
                end_str = "\n * "
            else:
                end_str = "\n \t * "

            print(
                f"in {year} _________________________________________ {np.round(self.vm.fuel_blend[fuel]['secondary']['share'][y] * 100)}%",
                end=end_str,
            )

    def add_carbon_dioxide_emissions(
        self, powertrain_short, fossil_co2, biogenic_co2
    ) -> None:
        idx = [f"transport, {self.vm.vehicle_type}, ", powertrain_short]

        array_idx = [
            d for d in self.array.coords["combined_dim"].values if powertrain_short in d
        ]

        _ = lambda x: np.where(x == 0, 1, x)

        self.A[
            :,
            self.inputs[("Carbon dioxide, fossil", ("air",), "kilogram")],
            self.find_input_indices(contains=tuple(idx)),
        ] = (
            self.array.sel(
                parameter="fuel mass",
                combined_dim=array_idx,
            )
            * fossil_co2
            / _(
                self.array.sel(
                    parameter=RANGE_PARAM[self.vm.vehicle_type],
                    combined_dim=array_idx,
                )
            )
            * -1
        )

        self.A[
            :,
            self.inputs[
                (
                    "Carbon dioxide, non-fossil",
                    ("air",),
                    "kilogram",
                )
            ],
            self.find_input_indices(
                contains=tuple(idx),
            ),
        ] = (
            self.array.sel(
                parameter="fuel mass",
                combined_dim=array_idx,
            )
            * biogenic_co2
            / _(
                self.array.sel(
                    parameter=RANGE_PARAM[self.vm.vehicle_type],
                    combined_dim=array_idx,
                )
            )
            * -1
        )

    def add_sulphur_emissions(self, fuel, powertrain_short, powertrains) -> None:
        # Fuel-based SO2 emissions
        # Sulfur concentration value for a given country, a given year, as concentration ratio

        sulfur_concentration = self.get_sulfur_content(self.vm.country, fuel)
        idx = [f"transport, {self.vm.vehicle_type}, ", powertrain_short]
        _ = lambda x: np.where(x == 0, 1, x)

        if sulfur_concentration:
            self.A[
                :,
                self.inputs[("Sulfur dioxide", ("air",), "kilogram")],
                self.find_input_indices(
                    contains=tuple(idx),
                    excludes=("battery",),
                ),
            ] = (
                self.array.sel(
                    parameter="fuel mass",
                    combined_dim=[
                        d
                        for d in self.array.coords["combined_dim"].values
                        if any(x in d for x in powertrains)
                    ],
                )
                / _(
                    self.array.sel(
                        parameter=RANGE_PARAM[self.vm.vehicle_type],
                        combined_dim=[
                            d
                            for d in self.array.coords["combined_dim"].values
                            if any(x in d for x in powertrains)
                        ],
                    )
                )
                * -1
                * sulfur_concentration
                * (64 / 32)  # molar mass of SO2/molar mass of O2
            )

    def add_fuel_to_vehicles(self, fuel, powertrains, powertrains_short) -> None:
        if [i for i in self.scope["powertrain"] if i in powertrains]:
            (
                fuel_blend_fossil_CO2,
                fuel_blend_biogenic_CO2,
            ) = self.get_fuel_blend_carbon_intensity(fuel)

            self.display_fuel_blend(fuel)

            # Fuel supply
            self.A[
                :,
                self.find_input_indices(
                    (f"fuel supply for {fuel} vehicles",),
                ),
                self.find_input_indices(
                    contains=(
                        f"transport, {self.vm.vehicle_type}, ",
                        powertrains_short,
                    ),
                    excludes=("battery",),
                ),
            ] = (
                (
                    self.array.sel(
                        parameter="fuel consumption",
                        combined_dim=[
                            d
                            for d in self.array.coords["combined_dim"].values
                            if any(x in d for x in powertrains)
                        ],
                    )
                    * self.array.sel(
                        parameter="fuel density per kg",
                        combined_dim=[
                            d
                            for d in self.array.coords["combined_dim"].values
                            if any(x in d for x in powertrains)
                        ],
                    )
                )
            ) * -1

            self.add_carbon_dioxide_emissions(
                powertrains_short,
                fuel_blend_fossil_CO2,
                fuel_blend_biogenic_CO2,
            )

            self.add_sulphur_emissions(fuel, powertrains_short, powertrains)

    def add_road_maintenance(self) -> None:
        # Infrastructure maintenance
        self.A[
            :,
            self.find_input_indices(("market for road maintenance",)),
            self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",)),
        ] = (
            1.29e-3 * -1
        )

    def add_road_construction(self) -> None:
        # Infrastructure
        self.A[
            :,
            self.find_input_indices(
                contains=("market for road",), excludes=("maintenance", "wear")
            ),
            self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",)),
        ] = (
            5.37e-7 * self.array.sel(parameter="driving mass") * -1
        )

    def add_exhaust_emissions(self) -> None:
        # Exhaust emissions
        # Non-fuel based emissions
        self.A[
            np.ix_(
                np.arange(self.iterations),
                [self.inputs[i] for i in self.exhaust_emissions],
                self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",)),
            )
        ] = (
            self.array.sel(parameter=list(self.exhaust_emissions.values())) * -1
        ).transpose(
            "value", "parameter", "combined_dim", "year"
        )

    def add_noise_emissions(self) -> None:
        # Noise emissions
        self.A[
            np.ix_(
                np.arange(self.iterations),
                [self.inputs[i] for i in self.noise_emissions],
                self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",)),
            )
        ] = (
            self.array.sel(parameter=list(self.noise_emissions.values())) * -1
        ).transpose(
            "value", "parameter", "combined_dim", "year"
        )

    def add_refrigerant_emissions(self) -> None:
        # Emissions of air conditioner refrigerant r134a
        # Leakage assumed to amount to 53g according to
        # https://treeze.ch/fileadmin/user_upload/downloads/Publications/Case_Studies/Mobility/544-LCI-Road-NonRoad-Transport-Services-v2.0.pdf
        # but only to cars with an AC system (meaning, with a cooling energy consumption)
        # and only for vehicles before 2022

        loss_rate = {
            "car": 0.75,
            "bus": 16,
            "truck": 0.94,
        }

        refill_rate = {
            "car": 0.55,
            "bus": 7.5,
            "truck": 1.1,
        }

        self.A[
            :,
            self.inputs[
                ("Ethane, 1,1,1,2-tetrafluoro-, HFC-134a", ("air",), "kilogram")
            ],
            self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",)),
        ] = (
            loss_rate[self.vm.vehicle_type]
            / self.array.sel(parameter="lifetime kilometers")
            * (self.array.sel(parameter="cooling energy consumption") > 0)
            * (np.array(self.scope["year"]) < 2022)
            * -1
        )

        self.A[
            :,
            self.find_input_indices(("market for refrigerant R134a",)),
            self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",)),
        ] = (
            (
                (loss_rate[self.vm.vehicle_type] + refill_rate[self.vm.vehicle_type])
                / self.array.sel(parameter="lifetime kilometers")
                * -1
            )
            * (self.array.sel(parameter="cooling energy consumption") > 0)
            * (np.array(self.scope["year"]) < 2022)
        )

    def add_abrasion_emissions(self) -> None:
        # Non-exhaust emissions

        abrasion_datasets = {
            (
                "road wear",
                "two-wheeler",
            ): "market for road wear emissions, passenger car",
            (
                "brake wear",
                "two-wheeler",
            ): "market for brake wear emissions, passenger car",
            (
                "tire wear",
                "two-wheeler",
            ): "market for tyre wear emissions, passenger car",
            ("road wear", "car"): "market for road wear emissions, passenger car",
            ("brake wear", "car"): "market for brake wear emissions, passenger car",
            ("tire wear", "car"): "market for tyre wear emissions, passenger car",
            ("road wear", "truck"): "treatment of road wear emissions, lorry",
            ("brake wear", "truck"): "treatment of brake wear emissions, lorry",
            ("tire wear", "truck"): "treatment of tyre wear emissions, lorry",
            ("road wear", "bus"): "treatment of road wear emissions, lorry",
            ("brake wear", "bus"): "treatment of brake wear emissions, lorry",
            ("tire wear", "bus"): "treatment of tyre wear emissions, lorry",
        }
        # Road wear emissions + 33.3% of re-suspended road dust
        self.A[
            :,
            self.find_input_indices(
                (abrasion_datasets[("road wear", self.vm.vehicle_type)],)
            ),
            self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",)),
        ] = self.array.sel(parameter="road wear emissions") + (
            0.333 * self.array.sel(parameter="road dust emissions")
        )

        self.A[
            :,
            self.find_input_indices(
                (abrasion_datasets[("tire wear", self.vm.vehicle_type)],)
            ),
            self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",)),
        ] = self.array.sel(parameter="tire wear emissions") + (
            0.333 * self.array.sel(parameter="road dust emissions")
        )

        # Brake wear emissions
        # BEVs only emit 20% of what a combustion engine vehicle emit according to
        # https://link.springer.com/article/10.1007/s11367-014-0792-4

        self.A[
            :,
            self.find_input_indices(
                (abrasion_datasets[("brake wear", self.vm.vehicle_type)],)
            ),
            self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",)),
        ] = self.array.sel(parameter="brake wear emissions") + (
            0.333 * self.array.sel(parameter="road dust emissions")
        )

    def remove_non_compliant_vehicles(self):
        """
        Remove vehicles from self.A that do not have a TtW energy superior to 0.
        """
        # Get the indices of the vehicles that are not compliant
        self.A = np.nan_to_num(self.A)
        idx = self.find_input_indices((f"{self.vm.vehicle_type.capitalize()}, ",))

        self.A[
            :,
            :,
            idx,
        ] *= (self.array.sel(parameter=["TtW energy"]) > 0).values
        self.A[:, idx, idx] = 1

        idx = self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",))

        self.A[
            :,
            :,
            idx,
        ] *= (self.array.sel(parameter=["TtW energy"]) > 0).values
        self.A[:, idx, idx] = 1

    def change_functional_unit(self) -> None:
        load_factor = self.get_load_factor()
        idx_cars = self.find_input_indices((f"transport, {self.vm.vehicle_type}, ",))
        idx_others = [i for i in range(self.A.shape[1]) if i not in idx_cars]

        self.A[
            np.ix_(
                np.arange(self.iterations),
                idx_others,
                idx_cars,
            )
        ] *= 1 / np.squeeze(load_factor).reshape(
            -1, len(idx_cars), len(self.scope["year"])
        )

        # iterate through self.inputs and change the unit
        keys_to_modify = {
            key: value
            for key, value in self.inputs.items()
            if key[0].startswith(f"transport, {self.vm.vehicle_type}")
        }

        for key, value in keys_to_modify.items():
            new_key = list(key)
            new_key[2] = self.func_unit
            del self.inputs[key]
            self.inputs[tuple(new_key)] = value

        # update self.rev_inputs
        self.rev_inputs = {v: k for k, v in self.inputs.items()}

    def export_lci(
        self,
        ecoinvent_version="3.9",
        filename=f"carculator_lci",
        directory=None,
        software="brightway2",
        format="bw2io",
    ):
        """
        Export the inventory. Can export to Simapro (as csv), or brightway2 (as bw2io object, file or string).
        :param db_name:
        :param ecoinvent_version: str. "3.5", "3.6", "3.7", "3.8" or "3.9"
        :param filename: str. Name of the file to be exported
        :param directory: str. Directory where the file is saved
        :param software: str. "brightway2" or "simapro"
        :param format: str. "bw2io" or "file" or "string"
        ::return: inventory, or the filepath where the file is saved.
        :rtype: list
        """

        if self.func_unit != "vkm":
            self.change_functional_unit()

        lci = ExportInventory(
            array=self.A,
            vehicle_model=self.vm,
            indices=self.rev_inputs,
            db_name=f"{filename}_{self.vm.vehicle_type}_{datetime.now().strftime('%Y%m%d')}",
        )

        if software == "brightway2":
            return lci.write_bw2_lci(
                ecoinvent_version=ecoinvent_version,
                directory=directory,
                filename=f"{filename}_{self.vm.vehicle_type}_{datetime.now().strftime('%Y%m%d')}",
                export_format=format,
            )

        else:
            return lci.write_simapro_lci(
                ecoinvent_version=ecoinvent_version,
                directory=directory,
                filename=f"{filename}_{self.vm.vehicle_type}_{datetime.now().strftime('%Y%m%d')}",
                export_format=format,
            )
