import itertools

import numpy as np
import pandas as pd
import xarray as xr

from .vehicle_input_parameters import VehicleInputParameters as vip


def fill_xarray_from_input_parameters(input_parameters, sensitivity=False, scope=None):
    """Create an `xarray` labeled array from the sampled input parameters.

    This function extracts the parameters' names and values contained in the
    `parameters` attribute of the :class:`CarInputParameters` class
    in :mod:`car_input_parameters` and insert them into a
    multi-dimensional numpy-like array from the *xarray* package
    (http://xarray.pydata.org/en/stable/).


    :param sensitivity:
    :param input_parameters: Instance of the :class:`TruckInputParameters` class
    in :mod:`truck_input_parameters`.
    :returns: `tuple`, `xarray.DataArray`
    - tuple (`size_dict`, `powertrain_dict`, `parameter_dict`, `year_dict`)
    - array

    Dimensions of `array`:

        0. Vehicle size, e.g. "3.5t", "7.5t", etc. str.
        1. Powertrain, e.g. "ICE-d", "BEV". str.
        2. Year. int.
        3. Samples.

    """

    # Check whether the argument passed is an instance of :class:`TruckInputParameters`
    if not isinstance(input_parameters, vip):
        raise TypeError(
            "The argument passed is not an object of the TruckInputParameter class"
        )

    if scope is None:
        scope = {
            "size": input_parameters.sizes,
            "powertrain": input_parameters.powertrains,
            "year": input_parameters.years,
        }
    else:
        if "size" not in scope:
            scope["size"] = input_parameters.sizes
        if "powertrain" not in scope:
            scope["powertrain"] = input_parameters.powertrains
        if "year" not in scope:
            scope["year"] = input_parameters.years

    # Make sure to include PHEV-e and PHEV-c-d if
    # PHEV-d is listed

    missing_pwts = [
        ("PHEV-d", "PHEV-e", "PHEV-c-d"),
        ("PHEV-p", "PHEV-e", "PHEV-c-p"),
    ]

    for missing_pwt in missing_pwts:
        if missing_pwt[0] in scope["powertrain"]:
            for p in missing_pwt[1:]:
                if not p in scope["powertrain"]:
                    scope["powertrain"].append(p)

    if any(s for s in scope["size"] if s not in input_parameters.sizes):
        raise ValueError("One of the size types is not valid.")

    if any(y for y in scope["year"] if y not in input_parameters.years):
        raise ValueError("One of the years defined is not valid.")

    if any(pt for pt in scope["powertrain"] if pt not in input_parameters.powertrains):
        raise ValueError("One of the powertrain types is not valid.")

    # if the purpose is not to do a sensitivity analysis
    # the dimension `value` of the array is as large as
    # the number of iterations to perform
    # that is, 1 in `static` mode, or several in `stochastic` mode.

    size_dict = {k: i for i, k in enumerate(scope["size"])}
    powertrain_dict = {k: i for i, k in enumerate(scope["powertrain"])}
    year_dict = {k: i for i, k in enumerate(scope["year"])}
    parameter_dict = {k: i for i, k in enumerate(input_parameters.parameters)}

    params = ["reference"] + input_parameters.input_parameters

    data_dict = [dict()]
    parameter_list = set()
    for param in input_parameters:
        pwt = (
            set(input_parameters.metadata[param]["powertrain"])
            if isinstance(input_parameters.metadata[param]["powertrain"], list)
            else set([input_parameters.metadata[param]["powertrain"]])
        )

        size = (
            set(input_parameters.metadata[param]["sizes"])
            if isinstance(input_parameters.metadata[param]["sizes"], list)
            else set([input_parameters.metadata[param]["sizes"]])
        )

        year = (
            set(input_parameters.metadata[param]["year"])
            if isinstance(input_parameters.metadata[param]["year"], list)
            else set([input_parameters.metadata[param]["year"]])
        )
        if (
            pwt.intersection(scope["powertrain"])
            and size.intersection(scope["size"])
            and year.intersection(scope["year"])
        ):
            powertrains = list(pwt.intersection(scope["powertrain"]))
            years = list(year.intersection(scope["year"]))
            sizes = list(size.intersection(scope["size"]))
            if len(sizes) > 1 and len(powertrains) > 1:
                pwt_size_couple = np.array(list(itertools.product(powertrains, sizes)))
                powertrains = pwt_size_couple[:, 0]
                sizes = pwt_size_couple[:, 1]

            data = {
                "size": sizes,
                "powertrain": powertrains,
                "parameter": input_parameters.metadata[param]["name"],
                "year": years,
                "data": input_parameters.values[param],
            }
            if not sensitivity:
                data["value"] = np.arange(input_parameters.iterations or 1)
            else:
                data["value"] = params

            data_dict.append(data)

            parameter_list.add(input_parameters.metadata[param]["name"])

    parameter_diff = parameter_list.symmetric_difference(input_parameters.parameters)

    for param in parameter_diff:
        data = {
            "size": scope["size"],
            "powertrain": scope["powertrain"],
            "parameter": param,
            "year": scope["year"],
            "data": 0.0,
        }
        if not sensitivity:
            data["value"] = np.arange(input_parameters.iterations or 1)
        else:
            data["value"] = params
        data_dict.append(data)

    df = pd.DataFrame.from_dict(data_dict)
    cols = ["powertrain", "size", "value", "year", "parameter"]
    df1 = pd.concat(
        [
            df[x]
            .explode()
            .to_frame()
            .assign(g=lambda x: x.groupby(level=0).cumcount())
            .set_index("g", append=True)
            for x in cols
        ],
        axis=1,
    )

    df = df.drop(cols, axis=1).join(df1.droplevel(1))
    df[cols] = df[cols].apply(lambda x: x.ffill())
    df.set_index(["size", "powertrain", "parameter", "year", "value"], inplace=True)
    df.dropna(inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    array = xr.DataArray.from_series(df["data"])
    array = array.astype("float32")
    array.coords["year"] = array.coords["year"].astype(int)
    array = array.fillna(0.0)

    if sensitivity:
        # we increase each value by 10% for each params excepting reference one

        for param in params[1:]:
            array.loc[dict(parameter=param, value=param)] *= 1.1

    return (size_dict, powertrain_dict, parameter_dict, year_dict), array
