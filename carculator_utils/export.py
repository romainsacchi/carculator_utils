"""
export.py contains the class Export, which offers methods to export the inventories
in different formats.
"""

import csv
import datetime
import io
import json
import os
import uuid
from typing import Dict, List, Tuple, Union

import bw2io
import numpy as np
import pyprind
import xarray as xr
import yaml
from bw2io.export.excel import create_valid_worksheet_name, safe_filename, xlsxwriter

from . import DATA_DIR, __version__


def load_inventories() -> list[dict]:
    """
    Load LCIs to fetch metadata from.
    """
    filepath = DATA_DIR / "lci" / "lci-premise_carculator_db.xlsx"
    if not filepath.is_file():
        raise FileNotFoundError("The database of inventories could not be found.")
    lci = bw2io.ExcelImporter(filepath)
    references = {
        lci.data[i]["name"]: {
            "source": lci.data[i].get("source"),
            "comment": lci.data[i].get("comment"),
        }
        for i in range(len(lci.data))
    }
    return references


def load_mapping(
    filename,
) -> Dict[Tuple[str, str, str, str], Tuple[str, str, str, str]]:
    """
    Load mapping dictionary between
    two versions of ecoinvent.
    """

    # Load the matching dictionary
    filepath = DATA_DIR / "export" / filename
    if not filepath.is_file():
        raise FileNotFoundError(
            "The dictionary of activities flows match " "could not be found."
        )
    with open(filepath, encoding="utf-8") as f:
        csv_list = [[val.strip() for val in r.split(";")] for r in f.readlines()]
    (_, _, *header), *data = csv_list

    dict_map = {}

    if "39" in filename:
        for row in data:
            (
                name_to,
                location_to,
                categories_to,
                unit_to,
                ref_prod_to,
                name_from,
                location_from,
                categories_from,
                unit_from,
                ref_prod_from,
            ) = row
            dict_map[
                (
                    name_to,
                    location_to,
                    tuple(categories_to.split("::")) if categories_to != "" else "",
                    unit_to,
                    ref_prod_to,
                )
            ] = (
                name_from,
                location_from,
                tuple(categories_from.split("::")) if categories_from != "" else "",
                unit_from,
                ref_prod_from,
            )
    else:
        for row in data:
            (
                name_to,
                location_to,
                unit_to,
                ref_prod_to,
                name_from,
                location_from,
                unit_from,
                ref_prod_from,
            ) = row
            dict_map[(name_to, location_to, unit_to, ref_prod_to)] = (
                name_from,
                location_from,
                unit_from,
                ref_prod_from,
            )

    return dict_map


def get_simapro_subcompartments() -> Dict[str, str]:
    # Load the matching dictionary between ecoinvent and Simapro subcompartments
    # contained in simapro_subcompartments.yaml

    filename = "simapro_subcompartments.yaml"
    filepath = DATA_DIR / "export" / filename
    if not filepath.is_file():
        raise FileNotFoundError(
            "The dictionary of subcompartments match "
            "between ecoinvent and Simapro could not be found."
        )

    # read YAML file
    with open(filepath, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return data


def get_simapro_biosphere() -> Dict[str, str]:
    # Load the matching dictionary between ecoinvent and Simapro biosphere flows
    # for each ecoinvent biosphere flow name, it gives the corresponding Simapro name

    filename = "simapro-biosphere.json"
    filepath = DATA_DIR / "export" / filename
    if not filepath.is_file():
        raise FileNotFoundError(
            "The dictionary of biosphere flow match "
            "between ecoinvent and Simapro could not be found."
        )
    with open(filepath, encoding="utf-8") as json_file:
        data = json.load(json_file)
    dict_bio = {}
    for d in data:
        dict_bio[d[2]] = d[1]

    return dict_bio


def get_simapro_technosphere() -> Dict[Tuple[str, str], str]:
    # Load the matching dictionary between ecoinvent and Simapro product flows

    filename = "simapro-technosphere-3.5.csv"
    filepath = DATA_DIR / "export" / filename
    with open(filepath, encoding="utf-8") as f:
        csv_list = [[val.strip() for val in r.split(";")] for r in f.readlines()]
    (_, _, *header), *data = csv_list

    dict_tech = {}
    for row in data:
        name, location, simapro_name = row
        simapro_name = simapro_name.split("|")[:2]
        dict_tech[(name, location)] = "|".join(simapro_name)

    return dict_tech


def rename_mapping(filename: str) -> Dict[str, str]:
    """
    Load the file rename_powertrains.yaml and return a dictionary
    """
    with open(DATA_DIR / "export" / filename, encoding="utf-8") as f:
        rename_map = yaml.safe_load(f)

    return rename_map


class ExportInventory:
    """
    Export the inventory to various formats

    """

    def __init__(
        self, array, vehicle_model, indices, db_name="carculator_utils export"
    ):
        self.array: xr.DataArray = array
        self.indices: Dict[int, Tuple[str, str, str, str]] = indices
        self.vm = vehicle_model
        self.rename_pwt = rename_mapping("rename_powertrains.yaml")
        self.rename_parameters = rename_mapping("rename_parameters.yaml")
        self.rename_vehicles()
        self.rev_rename_pwt = {v: k for k, v in self.rename_pwt.items()}
        self.db_name: str = db_name
        self.references = load_inventories()

        self.flow_map = {
            "3.5": load_mapping(filename="ei37_to_ei35.csv"),
            "3.6": load_mapping(filename="ei37_to_ei36.csv"),
            "3.7": load_mapping(filename="ei38_to_ei37.csv"),
            "3.7.1": load_mapping(filename="ei38_to_ei37.csv"),
            "3.9": load_mapping(filename="ei38_to_ei39.csv"),
        }

    def rename_vehicles(self) -> None:
        """
        Rename powertrain acronyms to full length descriptive terms

        """

        for k, value in self.indices.items():
            for key, val in self.rename_pwt.items():
                if key in value[0]:
                    new_val = list(value)
                    new_val[0] = new_val[0].replace(key, val)
                    self.indices[k] = tuple(
                        new_val,
                    )

    def write_lci(self, ecoinvent_version: str, year: int) -> List[Dict]:
        """
        Return the inventory as a dictionary
        If there are several values for one exchange, uncertainty information is generated.
        If `presamples` is True, returns the inventory as well as a `presamples` matrix.
        If `presamples` is False, returns the inventory with characterized uncertainty information.

        :returns: a dictionary that contains all the exchanges
        :rtype: dict
        """

        blacklist = {
            "3.5": [
                "latex production",
            ]
        }

        idx_year = self.vm.array.coords["year"].values.tolist().index(year)

        list_act = []

        # List of coordinates for non-zero values
        non_zeroes = np.nonzero(self.array[0, :, :, idx_year])
        # List of coordinates where activities present more than once
        # (to filter out "empty" activities, that is,
        # activities with only one reference product exchange)
        u, c = np.unique(non_zeroes[1], return_counts=True)
        dup = u[c > 1]

        # Filter out coordinates of "empty" activities
        coords = np.column_stack(
            (
                non_zeroes[0][np.isin(non_zeroes[1], dup)],
                non_zeroes[1][np.isin(non_zeroes[1], dup)],
            )
        )

        # Iterate through activities
        bar = pyprind.ProgBar(len(dup))
        for d in dup:
            bar.update(item_id=d)
            list_exc = []
            for row, col in coords[coords[:, 1] == d]:
                tuple_output = self.indices[col]
                tuple_input = self.indices[row]
                mult_factor = 1

                if tuple_output[0] in blacklist.get(ecoinvent_version, []):
                    continue

                if len(self.array[:, row, col, idx_year]) == 1:
                    # No uncertainty, only one value
                    amount = self.array[0, row, col, idx_year] * mult_factor

                else:
                    raise ValueError(
                        "Inventory export not "
                        "implemented for "
                        "stochastic analyses."
                    )

                exc = {
                    "name": tuple_input[0],
                    "unit": tuple_input[2],
                    "amount": amount * -1,
                }

                if len(tuple_input) == 3:
                    # biosphere exchange
                    exc["type"] = "biosphere"
                    exc["database"] = "biosphere3"
                    exc["categories"] = tuple_input[1]

                else:
                    exc["location"] = tuple_input[1]
                    exc["reference product"] = tuple_input[3]
                    exc["database"] = self.db_name

                    if tuple_output == tuple_input:
                        # reference product exchange
                        exc["amount"] *= -1
                        exc["type"] = "production"
                    else:
                        exc["type"] = "technosphere"

                list_exc.append(exc)

            source, comment = None, None

            if tuple_output[0] in self.references:
                source = self.references[tuple_output[0]].get("source")
                comment = self.references[tuple_output[0]].get("comment")
            elif self.vm.vehicle_type in tuple_output[0].lower():
                pass
            elif tuple_output[0].startswith("fuel supply for") or tuple_output[
                0
            ].startswith("electricity supply for"):
                pass
            else:
                print(f"Missing reference for {tuple_output[0]}")

            string = ""

            if f"{self.vm.vehicle_type}, " in tuple_output[0].lower():
                available_powertrains = [
                    self.rename_pwt[p] for p in self.vm.array.powertrain.values.tolist()
                ]
                available_sizes = self.vm.array.coords["size"].values.tolist()

                if any([w in tuple_output[0] for w in available_powertrains]) and any(
                    [w in tuple_output[0] for w in available_sizes]
                ):
                    possible_pwt = [
                        w for w in available_powertrains if w in tuple_output[0]
                    ]
                    if len(possible_pwt) > 1:
                        pwt = max(possible_pwt, key=len)
                    else:
                        pwt = possible_pwt[0]
                    pwt = self.rev_rename_pwt[pwt]

                    possible_sizes = [
                        w for w in available_sizes if w in tuple_output[0]
                    ]
                    if len(possible_sizes) > 1:
                        size = max(possible_sizes, key=len)
                    else:
                        size = possible_sizes[0]

                    string += f"Manufacture year: {year}. "
                    for param, formatting in self.rename_parameters.items():
                        if param not in self.vm.array.parameter.values:
                            continue

                        val = self.vm.array.sel(
                            powertrain=pwt,
                            size=size,
                            year=int(year),
                            value=0,
                            parameter=param,
                        ).values.astype(float)

                        if formatting.get("percentage", False):
                            val *= 100
                            val = "{:0.1f}".format(val)
                        else:
                            if val < 10:
                                val = "{:0.1f}".format(val)
                            else:
                                val = int(val)

                        string += f"{formatting['name']}: {val} {formatting['unit']}. "

            new_act = {
                "production amount": 1,
                "database": self.db_name,
                "name": tuple_output[0],
                "unit": tuple_output[2],
                "location": tuple_output[1],
                "exchanges": list_exc,
                "reference product": tuple_output[3],
                "type": "process",
                "code": str(uuid.uuid1()),
            }

            if source is not None:
                new_act["source"] = source
            if comment is not None:
                new_act["comment"] = comment
            elif string != "":
                new_act["comment"] = string
            else:
                pass

            list_act.append(new_act)

        return list_act

    def format_data_for_lci_for_bw2(self, data: List[dict]) -> List[dict]:
        rows = []
        rows.extend((["Database", self.db_name], ("format", "Excel spreadsheet")))
        rows.append([])

        for k in data:
            if k.get("exchanges"):
                rows.extend(
                    (
                        ["Activity", k["name"]],
                        ("location", k["location"]),
                        ("production amount", float(k["production amount"])),
                        ("reference product", k.get("reference product")),
                        ("type", "process"),
                        ("unit", k["unit"]),
                        ("source", k.get("source")),
                        ("comment", k.get("comment")),
                        ["Exchanges"],
                        [
                            "name",
                            "amount",
                            "database",
                            "location",
                            "unit",
                            "categories",
                            "type",
                            "reference product",
                            "tag",
                        ],
                    )
                )

                for e in k["exchanges"]:
                    rows.append(
                        [
                            e["name"],
                            float(e["amount"]),
                            e["database"],
                            e.get("location", "None"),
                            e["unit"],
                            "::".join(e.get("categories", ())),
                            e["type"],
                            e.get("reference product"),
                            e.get("tag", "other"),
                        ]
                    )
            else:
                rows.extend(
                    (
                        ["Activity", k["name"]],
                        ("type", "biosphere"),
                        ("unit", k["unit"]),
                        ("worksheet name", "None"),
                    )
                )
            rows.append([])

        return rows

    def format_data_for_lci_for_simapro(
        self, data: List[Dict], ei_version: str
    ) -> List[List]:
        # not all biosphere flows exist in simapro
        # load list from `simapro_blacklist.yaml`
        with open(
            DATA_DIR / "export" / "simapro_blacklist.yaml", "r", encoding="utf-8"
        ) as f:
            blacklist = yaml.safe_load(f)

        # load fields list from `simapro_fields.yaml`
        with open(
            DATA_DIR / "export" / "simapro_fields.yaml", "r", encoding="utf-8"
        ) as f:
            fields = yaml.safe_load(f)

        dict_tech = get_simapro_technosphere()
        dict_bio = get_simapro_biosphere()
        simapro_subs = get_simapro_subcompartments()

        rows = []

        for item in fields["headers"]:
            if item.startswith("{date"):
                item = item.replace(
                    "date", datetime.datetime.today().strftime("%d/%m/%Y")
                )
            rows.append([item])
        rows.append([])

        list_own_datasets = []

        for a in data:
            list_own_datasets.append(
                f"{a['name'].capitalize()} {{{a.get('location', 'GLO')}}})"
            )

        # We loop through the activities
        for a in data:
            # We fetch the main and sub categories (sub category is in fact a path)
            comment, source = None, None
            if a["name"] in self.references:
                source = self.references.get(a["name"]).get("source")
                comment = self.references.get(a["name"]).get("comment")

            main_category = (
                "waste treatment"
                if any(
                    i.lower() in a["name"].lower()
                    for i in (
                        "waste",
                        "emissions",
                        "treatment",
                        "scrap",
                        "used powertrain",
                        "disposal",
                        "sludge",
                        "used li-ion",
                    )
                )
                else "process"
            )
            category = "carculator"

            # We loop through the fields SimaPro expects to see
            for item in fields["fields"]:
                # If it is a waste treatment activity, we skip the field `Products`
                if main_category == "waste treatment" and item == "Products":
                    continue

                # It is not a waste treatment activity, we skip the field `Waste treatment`
                if main_category != "waste treatment" and item == "Waste treatment":
                    continue

                rows.append([item])

                if item == "Process name":
                    dataset_name = f"{a['name'].capitalize()} {{{a.get('location', 'GLO')}}} | Cut-off U"
                    rows.append([dataset_name])

                if item == "Type":
                    rows.append(["Unit process"])

                if item == "Comment":
                    string = ""
                    if comment is not None:
                        string = f"{a['comment']}. "

                    string += f"Originally published in: {source}. "

                    rows.append([string])

                if item == "Category type":
                    rows.append([main_category])

                if item == "Generator":
                    rows.append([f"carculator: {__version__}"])

                if item == "Geography":
                    rows.append([a["location"]])

                if item == "Time Period":
                    rows.append(["Refer to vehicle year."])

                if item == "Date":
                    rows.append([f"{datetime.datetime.today():%d.%m.%Y}"])

                if item in (
                    "Cut off rules",
                    "Capital goods",
                    "Technology",
                    "Representativeness",
                    "Boundary with nature",
                ):
                    rows.append(["Unspecified"])

                if item == "Infrastructure":
                    rows.append(["Yes"])

                if item == "External documents":
                    rows.append(["https://carculator.psi.ch"])

                if item in "System description":
                    rows.append(["carculator"])

                if item in "Allocation rules":
                    rows.append(
                        [
                            "In the instance of joint-production, allocation of process burden based on"
                            "economic relative revenue of each co-product."
                        ]
                    )

                if item == "Literature references":
                    rows.append(["Sacchi et al. 2022"])

                if item == "Collection method":
                    rows.append(
                        [
                            "Modeling and assumptions: https://carculator.readthedocs.io/en/latest/modeling.html"
                        ]
                    )

                if item == "Verification":
                    rows.append(["Peer-reviewed, but susceptible to change."])

                if item == "Waste treatment":
                    rows.append(
                        [
                            dict_tech.get((a["name"], a["location"]), dataset_name),
                            fields["unit"][a["unit"]],
                            1.0,
                            "not defined",
                            category,
                        ]
                    )

                if item == "Products":
                    for e in a["exchanges"]:
                        if e["type"] == "production":
                            rows.append(
                                [
                                    dict_tech.get(
                                        (a["name"], a["location"]), dataset_name
                                    ),
                                    fields["unit"][a["unit"]],
                                    1.0,
                                    "100%",
                                    "not defined",
                                    category,
                                ]
                            )

                if item == "Materials/fuels":
                    for e in a["exchanges"]:
                        if e["type"] == "technosphere":
                            if (
                                not any(
                                    i.lower() in e["name"].lower()
                                    for i in (
                                        "waste",
                                        "emissions",
                                        "treatment",
                                        "scrap",
                                        "used powertrain",
                                        "disposal",
                                        "sludge",
                                        "used li-ion",
                                        "mineral oil storage",
                                    )
                                )
                                or any(
                                    i in e["name"]
                                    for i in [
                                        "from municipal waste incineration",
                                        "municipal solid waste, incineration",
                                        "Biomethane",
                                        "biogas upgrading",
                                        "anaerobic digestion, with biogenic carbon uptake",
                                    ]
                                )
                                or any(
                                    i.lower() in e["reference product"].lower()
                                    for i in [
                                        "electricity",
                                    ]
                                )
                            ):
                                tupled = (
                                    e["name"],
                                    e.get("location", "GLO"),
                                    e["unit"],
                                    e["reference product"],
                                )

                                (
                                    e["name"],
                                    e["location"],
                                    e["unit"],
                                    e["reference product"],
                                ) = self.flow_map.get(ei_version, {tupled: tupled}).get(
                                    tupled, tupled
                                )

                                exchange_name = f"{e['name'].capitalize()} {{{e.get('location', 'GLO')}}}"

                                if exchange_name not in list_own_datasets:
                                    exchange_name = f"{e['reference product'].capitalize()} {{{e.get('location', 'GLO')}}}"

                                    if "market" in e["name"]:
                                        exchange_name += f"| market for {e['reference product'].lower()}"

                                    if "market group" in e["name"]:
                                        exchange_name += f"| market group for {e['reference product'].lower()}"

                                    if "production" in e["name"]:
                                        if len(e["reference product"].split(", ")) > 1:
                                            exchange_name += f"| {e['reference product'].split(', ')[0].lower()} production, "
                                            exchange_name += f"{e['reference product'].split(', ')[1].lower()}"

                                rows.append(
                                    [
                                        f"{dict_tech.get((e['name'], e['location']), exchange_name)} | Cut-off, U",
                                        fields["unit"][e["unit"]],
                                        "{:.3E}".format(e["amount"]),
                                        "undefined",
                                        0,
                                        0,
                                        0,
                                    ]
                                )

                if item == "Resources":
                    for e in a["exchanges"]:
                        if (
                            e["type"] == "biosphere"
                            and e["categories"][0] == "natural resource"
                        ):
                            if e["name"] not in blacklist:
                                rows.append(
                                    [
                                        dict_bio.get(e["name"], e["name"]),
                                        "",
                                        fields["unit"][e["unit"]],
                                        "{:.3E}".format(e["amount"]),
                                        "undefined",
                                        0,
                                        0,
                                        0,
                                    ]
                                )

                if item == "Emissions to air":
                    for e in a["exchanges"]:
                        if (
                            e["type"] == "biosphere" and e["categories"][0] == "air"
                        ) or e["name"] in [
                            "Carbon dioxide, from soil or biomass stock",
                            "Carbon dioxide, to soil or biomass stock",
                        ]:
                            if e["name"] not in blacklist:
                                if e["name"].lower() == "water":
                                    e["unit"] = "kilogram"
                                    e["amount"] /= 1000

                                if e["name"] in [
                                    "Carbon dioxide, to soil or biomass stock"
                                ]:
                                    rows.append(
                                        [
                                            dict_bio.get(e["name"], e["name"]),
                                            "",
                                            fields["unit"][e["unit"]],
                                            "{:.3E}".format(e["amount"] * -1),
                                            "undefined",
                                            0,
                                            0,
                                            0,
                                        ]
                                    )

                                else:
                                    rows.append(
                                        [
                                            dict_bio.get(e["name"], e["name"]),
                                            "",
                                            fields["unit"][e["unit"]],
                                            "{:.3E}".format(e["amount"]),
                                            "undefined",
                                            0,
                                            0,
                                            0,
                                        ]
                                    )

                if item == "Emissions to water":
                    for e in a["exchanges"]:
                        if e["type"] == "biosphere" and e["categories"][0] == "water":
                            if e["name"] not in blacklist:
                                if e["name"].lower() == "water":
                                    e["unit"] = "kilogram"
                                    e["amount"] /= 1000

                                rows.append(
                                    [
                                        dict_bio.get(e["name"], e["name"]),
                                        "",
                                        fields["unit"][e["unit"]],
                                        "{:.3E}".format(e["amount"]),
                                        "undefined",
                                        0,
                                        0,
                                        0,
                                    ]
                                )

                if item == "Emissions to soil":
                    for e in a["exchanges"]:
                        if e["type"] == "biosphere" and e["categories"][0] == "soil":
                            if e["name"] not in blacklist:
                                if len(e["categories"]) > 1:
                                    sub_compartment = simapro_subs[e["categories"][1]]
                                else:
                                    sub_compartment = ""

                                rows.append(
                                    [
                                        dict_bio.get(e["name"], e["name"]),
                                        "",
                                        fields["unit"][e["unit"]],
                                        "{:.3E}".format(e["amount"]),
                                        "undefined",
                                        0,
                                        0,
                                        0,
                                    ]
                                )

                if item == "Waste to treatment":
                    for e in a["exchanges"]:
                        is_waste = False
                        if e["type"] == "technosphere":
                            # We check if this is indeed a waste treatment activity
                            if e["name"] in self.references:
                                if self.references[e["name"]] == "waste treatment":
                                    is_waste = True
                            else:
                                if (
                                    any(
                                        i.lower() in e["name"].lower()
                                        for i in (
                                            " waste ",
                                            "emissions",
                                            "treatment",
                                            "scrap",
                                            "used powertrain",
                                            "used passenger car",
                                            "used electric passenger car",
                                            "municipal solid waste",
                                            "disposal",
                                            "rainwater mineral oil",
                                            "sludge",
                                            "used li-ion",
                                        )
                                    )
                                    and not any(
                                        i.lower() in e["name"].lower()
                                        for i in (
                                            "anaerobic",
                                            "cooking",
                                            "heat",
                                            "manual dismantling",
                                        )
                                    )
                                    and e["unit"] not in ["kilowatt hour", "megajoule"]
                                ):
                                    is_waste = True

                            # Yes, it is a waste treatment activity
                            if is_waste:
                                # In SimaPro, waste inputs are positive numbers
                                if e["amount"] < 0:
                                    e["amount"] *= -1

                                tupled = (
                                    e["name"],
                                    e.get("location", "GLO"),
                                    e["unit"],
                                    e["reference product"],
                                )

                                (
                                    e["name"],
                                    e["location"],
                                    e["unit"],
                                    e["reference product"],
                                ) = self.flow_map.get(ei_version, {tupled: tupled}).get(
                                    tupled, tupled
                                )

                                dataset_name = dict_tech.get(
                                    (e["name"], e["location"]),
                                    f"{e['name']} {{e['location']}}",
                                )

                                rows.append(
                                    [
                                        f"{dataset_name} | Cut-off, U",
                                        fields["unit"][e["unit"]],
                                        "{:.3E}".format(e["amount"]),
                                        "undefined",
                                        0,
                                        0,
                                        0,
                                    ]
                                )

                rows.append([])

        # System description
        rows.append(["System description"])
        rows.append([])
        rows.append(["Name"])
        rows.append(["carculator_utils"])
        rows.append([])
        rows.append(["Category"])
        rows.append(["transport"])
        rows.append([])
        rows.append(["Description"])
        rows.append(
            [
                "Prospective life cycle assessment model "
                "for vehicles developed by the Paul Scherrer Institute."
            ]
        )
        rows.append([])
        rows.append(["Cut-off rules"])
        rows.append(
            [
                "All environmentally-relevant flows are included, as far as the authors knowledge permits."
                "Also, residual material (e.g., biomass residue) and energy (e.g., waste heat) "
                "come free of burden, except for the necessary steps to make it reusable"
                " (transport, conditioning, etc.)."
            ]
        )
        rows.append([])
        rows.append(["Energy model"])
        rows.append(
            [
                "The energy consumption of vehicles calculated based on a physics model, including "
                "inertia, rolling resistance, aerodynamic drag, road gradient, etc."
            ]
        )
        rows.append([])
        rows.append(["Transport model"])
        rows.append(["Based on Sacchi et al. 2022"])
        rows.append([])
        rows.append(["Allocation rules"])
        rows.append(
            [
                "The system modeling is attributional. In the instance of joint-production, the allocation of "
                "burden between co-products is generally based on the relative economic revenue of "
                "each product, to align with the underlying database ecoinvent cut-off."
            ]
        )
        rows.append(["End"])
        rows.append([])

        # Literature reference
        rows.append(["Literature reference"])
        rows.append([])
        rows.append(["Name"])
        rows.append(["Sacchi et al. 2022"])
        rows.append([])
        rows.append(["Documentation link"])
        rows.append(["https://doi.org/10.1016/j.rser.2022.112475"])
        rows.append([])
        rows.append(["Comment"])
        rows.append(["Study available at: https://doi.org/10.1016/j.rser.2022.112475"])
        rows.append([])
        rows.append(["Category"])
        rows.append(["carculator_utils"])
        rows.append([])
        rows.append(["Description"])
        description = "When, where and how can the electrification of passenger cars reduce greenhouse gas emissions?"
        description += (
            "Romain Sacchi, Christian Bauer, Brian L. Cox and Chris L. Mutel\n"
        )
        description += "Renewable and Sustainable Energy Reviews, 2022"

        rows.append([description])

        return rows

    def get_export_filepath(self, filename, directory=None):
        # check that filepath exists
        directory = directory or os.getcwd()
        if not os.path.exists(directory):
            os.makedirs(directory)

        return os.path.join(directory, filename)

    def write_simapro_lci(
        self,
        ecoinvent_version: str,
        directory: str = None,
        filename: str = None,
        export_format: str = "file",
    ):
        for year in self.vm.array.coords["year"].values:
            filename = filename or safe_filename(
                f"carculator_export_{datetime.date.today()}"
            )

            filename += f"_{year}_simapro.csv"

            filepath_export = self.get_export_filepath(filename, directory)

            list_act = self.write_lci(
                ecoinvent_version=ecoinvent_version,
                year=year,
            )

            rows = self.format_data_for_lci_for_simapro(
                data=list_act, ei_version=ecoinvent_version
            )

            if export_format == "file":
                with open(filepath_export, "w", newline="", encoding="utf8") as csvFile:
                    writer = csv.writer(csvFile, delimiter=";")
                    for row in rows:
                        writer.writerow(row)
                csvFile.close()
                return filepath_export

            # string format
            csvFile = io.StringIO()
            writer = csv.writer(
                csvFile,
                delimiter=";",
                quoting=csv.QUOTE_NONE,
                quotechar="",
                escapechar="\\",
            )
            for row in rows:
                writer.writerow(row)
            csvFile.seek(0)
            return csvFile.read()

    def write_bw2_lci(
        self,
        ecoinvent_version: str,
        directory: str = None,
        filename: str = None,
        export_format: str = "file",
    ) -> Union[bytes, str, bw2io.importers.base_lci.LCIImporter]:
        """
        Export a file that can be consumed by the software defined in
        `software_compatibility`.
        Alternatively, exports a string representation of the file
        (in case the invenotry should be downloaded
        from a browser, for example)

        :param vehicle_specs:
        :param filename:
        :param export_format: file, string, bw2io
        :param directory: str. path to export the file to.
        :type directory: str or pathlib.Path
        :param ecoinvent_version: str. "3.5", "3.6", "3.7" or "3.8"
        :type ecoinvent_version: str

        If "string", returns a string.

        :returns: returns the file path of the exported inventory.
        :rtype: str
        """

        importers = []

        for year in self.vm.array.coords["year"].values:
            file = filename or safe_filename(
                f"carculator_export_{datetime.date.today()}"
            )
            file += f"_{year}_bw2.xlsx"

            filepath_export = self.get_export_filepath(file, directory)

            data = self.write_lci(ecoinvent_version=ecoinvent_version, year=year)

            if export_format == "bw2io":
                lci = bw2io.importers.base_lci.LCIImporter(self.db_name)
                lci.data = data
                lci.db_name = self.db_name
                return lci

            formatted_data = self.format_data_for_lci_for_bw2(data)
            output = io.BytesIO() if export_format == "string" else filepath_export
            workbook = xlsxwriter.Workbook(output, {"in_memory": True})

            bold = workbook.add_format({"bold": True})
            bold.set_font_size(12)
            highlighted = {
                "Activity",
                "Database",
                "Exchanges",
                "Parameters",
                "Database parameters",
                "Project parameters",
            }
            frmt = lambda x: bold if row[0] in highlighted else None
            sheet = workbook.add_worksheet(create_valid_worksheet_name("inventories"))

            for row_index, row in enumerate(formatted_data):
                for col_index, value in enumerate(row):
                    if value is None:
                        continue
                    elif isinstance(value, float):
                        sheet.write_number(row_index, col_index, value, frmt(value))
                    else:
                        sheet.write_string(row_index, col_index, value, frmt(value))

            if export_format == "file":
                workbook.close()
            else:
                # return string
                workbook.close()
                output.seek(0)
                importers.append(output.read())

        if export_format == "file":
            return filepath_export
        return importers
