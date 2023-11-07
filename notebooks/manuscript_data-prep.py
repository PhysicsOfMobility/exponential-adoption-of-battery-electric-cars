# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python (gendev)
#     language: python
#     name: gendev
# ---

# # Prepare data for the article "Exponential adoption of battery electric cars" by Felix Jung, Malte Schröder and Marc Timme

# Date convention: measurements taken at the **end** of the calendar year.
#
# This notebook takes raw data acquired from the sources referenced below and in the manuscript and uses them to create five output files:
#
# - `regions.pq`
# - `bec-stock-by-region.pq`
# - `bec-share-by-region.pq`
# - `pc-stock-by-region-europe-us.pq`
# - `pc-stock-by-region-world.pq`
#
# The first is just an overview over the available data, the other four are used by the notebook `manuscript_figures.py` to produce the analysis resulting in the manuscript's figures and tables.
#
# To get started, you must first acquire the data from the following sources:
#
# **[IEA](https://www.iea.org/data-and-statistics/data-product/global-ev-outlook-2023)**
#
# The dataset (GEVO 2022) used for the manuscript is not available anymore on the IEA's website. Instead, you may use the 2023 one:
#
# Procedure to retrieve the data:
#
# - Select "Global EV Data" > "EV data by country" > "CSV"
# - Store the file as `data/raw/IEA/23/IEA-EV-data.csv`
#
#
# **[Eurostat](https://ec.europa.eu/eurostat/databrowser/view/ROAD_EQS_CARPDA__custom_2967363/default/table?lang=en)**
#
# Procedure to retrieve the data:
#
# - Select "Download" > "Full dataset \[road_eqs_carpda\]" > "SDMX-CSV 1.0"
# - Store file as `data/raw/eurostat/22/road_eqs_carpda_linear.csv.gz`
# - Unzip the file
#    - `cd data/raw/eurostat/22/`
#    - `gunzip road_eqs_carpda_linear.csv.gz`
#
# **[FHWA](https://www.fhwa.dot.gov/policyinformation/statistics.cfm)**
#
# Procedure to retrieve the data:
#
# - For each year from 2007 to 2021, select the statistics publication from the dropdown menu
#     - Retrieve the following table: "State motor-vehicle registrations"
#     - Download the Excel version as 
#     - Store file as `data/raw/FHWA/<year>_motor_vehicles_us.{xls, xlsx}` 
#
# **[OICA](https://www.oica.net/category/vehicles-in-use/)**
#
# Procedure to retrieve the data:
#
# - Download "World Vehicles in use" > "By country/region and type 2015-2020" > "Passenger Cars (XLS)"
# - Store the file at `data/raw/OICA/20/PC-World-vehicles-in-use-2020.xlsx`
#
#
# **[Statistics Iceland](https://statice.is/statistics/environment/transport/vehicles/)**
#
# Procedure to retrieve the data:
#
# - "Vehicles and roads"
# - Registered motor vehicles 1950-2021
# - Select all years
# - Select "Passenger cars"
# - "Show table"
# - "Excel (xlsx)"
# - Store the xlsx file in `data/raw/iceland`

# +
# %matplotlib inline

import re
from copy import deepcopy
from fxutil.imports.general import *
from fxutil.plotting import SaveFigure, evf, figax, easy_prop_cycle
from fxutil import get_git_repo_path, scinum
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pprint import pprint
# -
data_dir = get_git_repo_path() / "data"
input_dir = data_dir / "raw"
output_dir = data_dir / "crunched"
output_dir.mkdir(parents=True, exist_ok=True)

# ## General Settings

# ### Regions

# +
developing_regions = {
    "Brazil",
    "Chile",
    "China",
    "India",
    "Mexico",
    "South Africa",
    "Bulgaria",
    "Kosovo",
    "North Macedonia",
    "Romania",
    "Turkey",
}

developed_regions = {
    "Australia",
    "Belgium",
    "Canada",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Iceland",
    "Italy",
    "Japan",
    "Korea",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Poland",
    "Portugal",
    "Spain",
    "Sweden",
    "Switzerland",
    "USA",
    "United Kingdom",
    "Austria",
    "Croatia",
    "Cyprus",
    "Czechia",
    "Estonia",
    "Hungary",
    "Ireland",
    "Latvia",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Slovakia",
    "Slovenia",
}


# +
regions = pd.DataFrame(index=sorted(developed_regions | developing_regions))

regions["is_country"] = True
regions.loc[[*developed_regions], "is_developed"] = True
regions.loc[[*developing_regions], "is_developing"] = True
regions.fillna(False, inplace=True)
# -
# ## Data sources

# ### IEA 
# https://www.iea.org/data-and-statistics/data-product/global-ev-outlook-2022#data-sets

# iea_dir = input_dir / "IEA" / "22"  # if using the GEVO 2022 dataset
iea_dir = input_dir / "IEA" / "23"  # if using the GEVO 2023 dataset

iea_df = pd.read_csv(iea_dir / "IEA-EV-data.csv")

# convert to beginning of the year:
iea_df.year += 1

iea_df = (
    iea_df.pivot(
        index="year",
        columns=[
            "region",
            "category",
            "mode",
            "parameter",
            "powertrain",
            "unit",
        ],
        values="value",
    )
    .sort_index(axis=0)
    .sort_index(axis=1)
)

# #### BEVs

bev_stock_share_by_region = iea_df.loc[
    :, np.s_[:, "Historical", "Cars", "EV stock share", "EV", "percent"]
].dropna(how="all")
bev_stock_share_by_region.columns = bev_stock_share_by_region.columns.droplevel(
    [*np.r_[1:6]]
)
bev_stock_share_by_region /= 100

bev_stock_by_region = iea_df.loc[
    :, np.s_[:, "Historical", "Cars", "EV stock", "BEV", "stock"]
].dropna(how="all")
bev_stock_by_region.columns = bev_stock_by_region.columns.droplevel([*np.r_[1:6]])

iea_regions = set(bev_stock_by_region.columns)


regions = regions.reindex(set(regions.index) | iea_regions)
regions.loc[[*iea_regions], "in_dataset_iea"] = True
regions["in_dataset_iea"].fillna(False, inplace=True)

# ### Eurostat (Europe)
# https://ec.europa.eu/eurostat/databrowser/view/ROAD_EQS_CARPDA__custom_2967363/default/table?lang=en

eurostat_dir = input_dir / "eurostat" / "22"

country_codes = (
    pd.read_csv(eurostat_dir / "geo_en.dic", sep="\t", index_col=0, header=None)
    .squeeze()
    .to_dict()
)

# +
euro_df = pd.read_csv(eurostat_dir / "road_eqs_carpda_linear.csv")
euro_df.drop(["DATAFLOW", "LAST UPDATE", "freq", "unit"], axis=1, inplace=True)

euro_df.rename(
    {
        "mot_nrg": "powertrain",
        "geo": "country_code",
        "TIME_PERIOD": "year",
        "OBS_VALUE": "stock",
        "OBS_FLAG": "definition_differs",
    },
    axis=1,
    inplace=True,
)

euro_df["country"] = euro_df.country_code.map(country_codes)
assert not any(euro_df.country.isna())

euro_df.country = euro_df.country.str.replace(" \(.*\)", "", regex=True)

euro_df["definition_differs"] = euro_df.definition_differs.map(
    {m.nan: False, "d": True}
)
assert not any(euro_df.definition_differs.isna())
# -

euro_df_wide = (
    euro_df.pivot(index="year", values="stock", columns=["country", "powertrain"])
    .sort_index(axis=0)
    .sort_index(axis=1)
)

euro_pc_stock_by_region = euro_df_wide.loc[:, np.s_[:, "TOTAL"]]
euro_pc_stock_by_region.columns = euro_pc_stock_by_region.columns.droplevel(1)

euro_pc_stock_last = euro_pc_stock_by_region.ffill().loc[2019].astype("u8")

euro_regions = set(euro_pc_stock_by_region.columns)

regions = regions.reindex(set(regions.index) | euro_regions)
regions.loc[[*euro_regions], "in_dataset_eurostat"] = True
regions["in_dataset_eurostat"].fillna(False, inplace=True)

# #### Side consideration: Look at powertrain technology dominance

euro_sum_df = euro_df_wide.groupby(
    euro_df_wide.columns.get_level_values(1), axis=1
).sum()


eurostat_powertrain_hmap = {
    "ALT": "alternative energy",
    "BIFUEL": "bi-fuel",
    "BIODIE": "biodiesel",
    "BIOETH": "bioethanol",
    "DIE": "diesel",
    "DIE_X_HYB": "diesel (excluding hybrids)",
    "ELC": "electricity",
    "ELC_DIE_HYB": "hybrid diesel-electric",
    "ELC_DIE_PI": "plug-in hybrid diesel-electric",
    "ELC_PET_HYB": "hybrid electric-petrol",
    "ELC_PET_PI": "plug-in hybrid petrol-electric",
    "GAS": "natural Gas",
    "HYD_FCELL": "hydrogen and fuel cells",
    "LPG": "liquefied petroleum gases (LPG)",
    "PET": "petroleum products",
    "PET_X_HYB": "petrol (excluding hybrids)",
    "OTH": "residual category",
    "TOTAL": "total",
}

euro_sum_df.rename(eurostat_powertrain_hmap, inplace=True, axis=1)

euro_pt_fractions = euro_sum_df.loc[:, [*set(euro_sum_df.columns) - {"total"}]].div(
    euro_sum_df.loc[:, "total"], axis=0
)

fig, ax = figax()
euro_pt_fractions.drop("alternative energy", axis=1).loc[2019].sort_values(
    ascending=False
).plot(kind="barh")
ax.set_xlabel(r"Eurostat powertrain fractions in 2019")
ax.set_xlim(1e-7, 1)
ax.set_xscale("log")

euro_pt_fractions[["electricity", "hydrogen and fuel cells", "bioethanol"]] * 100

# --> Hydrogen fleet size is consistently smaller than EV by three orders of magnitude (1/1000)

# ## Federal Highway Administration
# Source: https://www.fhwa.dot.gov/policyinformation/statistics.cfm

fhwa_dir = input_dir / "FHWA"

us_total_pc_dict = {}
for us_mv_fn in fhwa_dir.glob("*.xls*"):
    us_mv_path_path = fhwa_dir / us_mv_fn

    year = str(us_mv_fn.stem).split("_")[0]
    df = pd.read_excel(us_mv_path_path)
    us_total_pc_dict[int(year)] = (
        df[df.iloc[:, 0].str.strip() == "Total"].iloc[:, 3].squeeze()
    )
us_pc_stock = pd.Series(us_total_pc_dict)


regions.loc["USA", "in_dataset_fhwa"] = True
regions["in_dataset_fhwa"].fillna(False, inplace=True)

# ## International Organization of Motor Vehicle Manufacturers (OICA)
# Source: https://www.oica.net/category/vehicles-in-use/

oica_dir = input_dir / "OICA" / "20"

oica_world_pc_stock = pd.read_excel(
    oica_dir / "PC-World-vehicles-in-use-2020.xlsx", usecols="B:D", header=3
).dropna(how="all")

# +
oica_world_pc_stock.loc[1:33, "region"] = "Europe"
oica_world_pc_stock.loc[35:48, "region"] = "America"
oica_world_pc_stock.loc[50:70, "region"] = "Asia/Oceania/Middle East"
oica_world_pc_stock.loc[72:79, "region"] = "Africa"
oica_world_pc_stock.loc[81, "region"] = "All Countries/Regions"


oica_world_pc_stock.loc[2:26, "sub_region"] = "EU 27 countries + EFTA + UK"
oica_world_pc_stock.loc[27:33, "sub_region"] = "Russia, Turkey & other Europe"
oica_world_pc_stock.loc[36:39, "sub_region"] = "NAFTA"
oica_world_pc_stock.loc[40:48, "sub_region"] = "Central & South America"

oica_world_pc_stock.loc[[1, 35, 50, 72, 81], "type"] = "region_total"
oica_world_pc_stock.loc[[2, 27, 36, 40], "type"] = "sub_region_total"
oica_world_pc_stock.loc[[26, 33, 48, 70, 79], "type"] = "sub_region_remainder"

oica_world_pc_stock.loc[oica_world_pc_stock.loc[:, "type"] == "nan", "type"] = "country"

oica_world_pc_stock.rename(
    {"REGIONS/COUNTRIES": "oica_region_name", "2015": "stock_2015", 2020: "stock_2020"},
    axis=1,
    inplace=True,
)
# -



oica_world_pc_stock.loc[:, ["stock_2015", "stock_2020"]] *= 1000

# +
known_regions_lower = [*regions.index.str.lower()]
known_regions = [*regions.index]
special_cases_mapping = {
    "czech republic": "Czechia",
    "belarus": "Belarus",
    "russia": "Russia",
    "serbia": "Serbia",
    "ukraine": "Ukraine",
    "united states of america": "USA",
    "argentina": "Argentina",
    "colombia": "Colombia",
    "ecuador": "Ecuador",
    "peru": "Peru",
    "venezuela": "Venezuela",
    "indonesia": "Indonesia",
    "iran": "Iran",
    "iraq": "Iraq",
    "israel": "Israel",
    "kazakhstan": "Kazakhstan",
    "malaysia": "Malaysia",
    "pakistan": "Pakistan",
    "philippines": "Philippines",
    "south korea": "South Korea",
    "syria": "Syria",
    "taiwan": "Taiwan",
    "thailand": "Thailand",
    "united arab emirates": "UAE",
    "vietnam": "Vietnam",
    "algeria": "Algeria",
    "egypt": "Egypt",
    "libya": "Libya",
    "morocco": "Morocco",
    "nigeria": "Nigeria",
}

oica_region_mapping = {}

for i, oica_region in (
    oica_world_pc_stock.loc[oica_world_pc_stock.type == "country", "oica_region_name"]
    .str.lower()
    .items()
):
    try:
        mapped_region = known_regions[known_regions_lower.index(oica_region)]
    except ValueError:
        mapped_region = special_cases_mapping[oica_region]

    oica_region_mapping[i] = mapped_region
# -

oica_world_pc_stock["internal_region_name"] = pd.Series(oica_region_mapping)

# +
world_pc_stock = (
    oica_world_pc_stock[oica_world_pc_stock.type == "country"][
        ["stock_2015", "stock_2020", "internal_region_name"]
    ]
    .sort_values("internal_region_name")
    .set_index("internal_region_name")
)

world_pc_stock.index.rename("region", inplace=True)
world_pc_stock.columns = [2015, 2020]
world_pc_stock = world_pc_stock.T
# -
tmp_pc_values_world = oica_world_pc_stock.loc[
    oica_world_pc_stock.region == "All Countries/Regions", ["stock_2015", "stock_2020"]
].T
tmp_pc_values_world.index = [2015, 2020]
tmp_pc_values_world = tmp_pc_values_world.squeeze().rename("World")

# manually add world data as the only region total that we actually need
world_pc_stock = pd.concat([world_pc_stock, tmp_pc_values_world], axis=1)

oica_regions = {*oica_region_mapping.values()}
regions = regions.reindex(set(regions.index) | oica_regions)
regions.loc[[*world_pc_stock], "in_dataset_oica"] = True
regions["in_dataset_oica"].fillna(False, inplace=True)
regions["is_country"].fillna(False, inplace=True)
regions["in_dataset_iea"].fillna(False, inplace=True)
regions["in_dataset_eurostat"].fillna(False, inplace=True)
regions["in_dataset_fhwa"].fillna(False, inplace=True)
regions.sort_index(inplace=True)

# ## Passenger cars in Iceland
# Number at end of year

iceland_dir = input_dir / "iceland"

# +
iceland_pc_df = pd.read_excel(
    next(iceland_dir.glob("*.xlsx")),
    usecols="A:B",
    header=2,
    index_col=0,
    skipfooter=29,
)

iceland_pc_df.index += 1
# -

iceland_pc_df.tail()

euro_pc_stock_by_region = euro_pc_stock_by_region.combine_first(
    iceland_pc_df.rename({"Passenger cars": "Iceland"}, axis=1)
)

# ## Join and Export Data

# ### Regions overview

countries_index = list(set(regions.index) - {"World", "Other Europe"})
assert not euro_regions - developed_regions - developing_regions

regions.to_parquet(output_dir / "regions.pq")

# ### BEVs

bev_stock_by_region.to_parquet(output_dir / "bec-stock-by-region.pq")
bev_stock_share_by_region.to_parquet(output_dir / "bec-share-by-region.pq")

# ### Passenger Cars

pc_stock_by_region = euro_pc_stock_by_region.copy()
pc_stock_by_region = pc_stock_by_region.reindex(
    pc_stock_by_region.index.join(us_pc_stock.index, how="outer")
)
pc_stock_by_region["USA"] = us_pc_stock

pc_stock_by_region.to_parquet(output_dir / "pc-stock-by-region-europe-us.pq")

world_pc_stock.to_parquet(output_dir / "pc-stock-by-region-world.pq")


