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

# # Plot Figures and create LaTeX Tables for the article "Exponential adoption of battery electric cars" by Felix Jung, Malte Schröder and Marc Timme

# +
# %matplotlib inline

import inspect
import re
from tqdm import tqdm
import warnings
import subprocess
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Optional, Tuple

import scipy.stats as st
import uncertainties as unc
import uncertainties.unumpy as unp
from fxutil import get_git_repo_path, scinum
from fxutil.imports.general import *
from fxutil.plotting import SaveFigure, easy_prop_cycle, evf, figax, set_aspect
from mpl_toolkits.axes_grid1 import make_axes_locatable


# +
data_dir = get_git_repo_path() / "data"
crunched_dir = data_dir / "crunched"
plot_dir = data_dir / "figures"
tables_dir = data_dir / "tables"

plot_dir.mkdir(exist_ok=True, parents=True)
tables_dir.mkdir(exist_ok=True, parents=True)
# -

sf = SaveFigure(
    plot_dir,
    output_transparency=False,
    show_dark=False,
    name_str_space_replacement_char="-",
    filetypes="pdf",
)

# ## Settings
# ### PLOS ONE figure requirements
# As per https://journals.plos.org/plosone/s/figures#loc-dimensions

fig_width = 5.2  # in
fig_height_max = 8.75  # in

fig_dpi = 300
fig_axes_aspect = 3 / 4

# ### Label terms and plot colors

label_fleet_size = "Fleet size"
label_date = "Year"
label_BEC = "BEC"
label_PC = "Total PC"
label_BEC_stock = "BEC fleet size"
label_BEC_share = "BEC fraction"
label_BEC_share_percent = r"BEC fraction [$\%$]"

model_colors_dict = {"exp": "C7", "log": "C5", "bass": "C3"}
model_labels_dict = {"exp": "Exponential", "log": "Logistic", "bass": "Bass"}

# ## Load Data

# Worldwide BEC data (IEA)

bec_stock_by_region = pd.read_parquet(crunched_dir / "bec-stock-by-region.pq")
bec_share_by_region = pd.read_parquet(crunched_dir / "bec-share-by-region.pq")

# PC data Europe (Eurostat)

pc_stock_by_region_europe_us = pd.read_parquet(
    crunched_dir / "pc-stock-by-region-europe-us.pq"
)

# PC data worldwide (OICA)

pc_stock_by_region_world = pd.read_parquet(crunched_dir / "pc-stock-by-region-world.pq")

# combine data sources

pc_stock_by_region = (
    pd.concat(
        {
            "eurostat": pc_stock_by_region_europe_us[
                [*set(pc_stock_by_region_europe_us) - {"USA", "Iceland"}]
            ],
            "fhwa": pc_stock_by_region_europe_us["USA"],
            "iceland": pc_stock_by_region_europe_us["Iceland"],
            "oica": pc_stock_by_region_world,
        },
        names=["source", "region"],
        axis=1,
    )
    .swaplevel(axis=1)
    .sort_index(axis=1)
)

# Add "Other Europe" to the PC dataframe

# +
# source for definition of "Other Europe":
# https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer

countries_other_europe = [
    "Austria",
    "Bulgaria",
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
    "Romania",
    "Slovakia",
    "Slovenia",
    "Turkey",
]
# -

# For all regions w/ data for 2020 available from OICA, use the OICA value. Otherwise, use padded value from 2019 (eurostat).
for year in [2015, 2020]:
    df = pc_stock_by_region.ffill().loc[year, countries_other_europe].unstack()

    pc_stock_by_region.loc[year, ("Other Europe", "oica+eurostat")] = (
        df["oica"].fillna(df.eurostat).sum()
    )
    pc_stock_by_region.sort_index(axis=1, inplace=True)

# ## BEC Model

# ### BEC Model Class


class BEVModel:
    """
    Class representing a single BEV adoption model.
    """

    def __init__(
        self,
        ser: pd.Series,
        model: Callable,
        ref_pc_stock: int,
        ser_share: pd.Series | None = None,
        transformation: Optional[Tuple[Callable, Callable]] = None,
        alpha: float = 0.05,
        p: int = 0,
        guess: Tuple = None,
        bounds: Tuple = None,
        plot_label: str = None,
        plot_label_data=None,
        plot_color="C0",
    ):
        """
        Initialize BEVModel.

        Parameters
        ----------
        ser
            Historic BEC data to model, index must be single-level datetime or integer years.
        model
            Callable with time `x` being the first argument and `p` remaining arguments
            corresponding to the tunable parameters.
        ref_pc_stock
            Reference PC fleet size
        ser_share
            Historic BEC share data (for plotting). Optional.
        transformation
            Tuple of two transformation functions, forward and backward.
            The second (backward) function is applied to the data before
            fitting the model, the first (forward) function is applied to model prediction.
        alpha
            Certainty level for CI estimation.
        p
            Number of model parameters.
        guess
            Initial parameter guess for the model fitting process.
        bounds: Pair of iterables of size `p`
            Bounds constraining the parameters during the fitting process.
        plot_label
            Plot label for the model curve.
        plot_label_data
            Plot label for the data points.
        plot_color
            Model curve color
        """
        if transformation is None:
            self.ftrans = self.btrans = lambda x: x
        else:
            self.ftrans, self.btrans = transformation

        self.model = model

        self.ref_pc_stock = ref_pc_stock
        self.alpha = alpha
        self.p = p
        self.guess = guess
        self.bounds = bounds
        self.plot_label = plot_label
        self.plot_label_data = plot_label_data
        self.plot_color = plot_color

        self.ser = ser
        self.ser_share = ser_share

        if isinstance(ser.index, pd.DatetimeIndex):
            self.years = ser.index.year.values
        else:
            self.years = ser.index.values
        self.values = ser.values

        if ser_share is not None:
            if isinstance(ser_share.index, pd.DatetimeIndex):
                self.years_share = ser_share.index.year.values
            else:
                self.years_share = ser_share.index.values
            self.values_share = ser_share.values

        self.n = len(self.values)

    def fit(self, debug=False, year_range=None):
        """
        Perform the fit procedure.

        Parameters
        ----------
        debug
            Debug mode
        year_range
            Restrict fitting to year range
        """
        self.popt, self.pcov = scipy.optimize.curve_fit(
            ft.partial(self.model, unumpy=False),
            self.years
            if year_range is None
            else self.ser.loc[slice(*year_range)].index,
            self.btrans(
                self.values
                if year_range is None
                else self.ser.loc[slice(*year_range)].values
            ),
            p0=self.guess,
            maxfev=2000,
            bounds=self.bounds if self.bounds is not None else (-np.inf, np.inf),
        )

        self.coefs = unc.correlated_values(self.popt, self.pcov)

        if debug:
            print(f"{self.popt=}")
            print(f"{self.pcov=}")
            print(f"{self.coefs=}")

        self.nom = lambda x: unp.nominal_values(self.model(x, *self.coefs, unumpy=True))
        self.std = lambda x: unp.std_devs(self.model(x, *self.coefs, unumpy=True))

        perc = scipy.stats.norm.ppf(1 - self.alpha / 2)

        self.r2 = 1.0 - (
            sum((self.values - self.ftrans(self.nom(self.years).astype(float))) ** 2)
            / (self.n * np.var(self.values, ddof=0))
        )
        self.r2adj = 1 - (1 - self.r2) * (self.n - 1) / (self.n - self.p - 1)

        self.lcb = lambda x: self.nom(x) - perc * self.std(x)
        self.ucb = lambda x: self.nom(x) + perc * self.std(x)

        self.mae = np.mean(np.abs(self.ftrans(self.nom(self.years)) - self.values))
        self.rmsd = m.sqrt(
            np.mean((self.ftrans(self.nom(self.years)) - self.values) ** 2)
        )

    def plot(
        self, ax=None, ci=False, share=False, percent=False, t_min_max=None, **kwargs
    ):
        """
        Plot the fitted model curve.

        Parameters
        ----------
        ax
            Matplotlib axes to use
        ci
            Plot confidence intervals
        share
            If true, don't plot absolute fleet size, but share of total
        percent
            Plot share in percent
        t_min_max
            Plot time range
        """
        if ax is None:
            ax = plt.gca()

        if t_min_max is None:
            t_min, t_max = ax.get_xlim()
        else:
            t_min, t_max = t_min_max

        if share:
            if percent:
                share_denominator = self.ref_pc_stock / 100
            else:
                share_denominator = self.ref_pc_stock
        else:
            share_denominator = 1

        try:
            (main_line,) = ax.plot(
                *evf(
                    np.linspace(t_min, t_max, 200),
                    lambda x: self.ftrans(self.nom(x)) / share_denominator,
                ),
                **(dict(c=self.plot_color) | kwargs),
            )

        except (ValueError, OverflowError) as e:
            print(
                f"You told me to plot the ftransed nom in the range "
                f"({t_min}, {t_max}), which failed. Error was {e}."
            )
            raise

        if ci:
            ax.fill_between(
                np.linspace(t_min, t_max, 200),
                self.ftrans(self.lcb(np.linspace(t_min, t_max, 200))),
                self.ftrans(self.ucb(np.linspace(t_min, t_max, 200))),
                color=main_line.get_color(),
                alpha=0.2,
                lw=0.7,
                edgecolor="k",
            )

    def get_data(
        self,
        share=False,
        percent=False,
        infer_share_from_last_pc_stock=False,
    ):
        """
        Returns the historic BEC data.

        Parameters
        ----------
        share
            Return the BEC share instead of absolute numbers
        percent
            If returning share, return values in percent
        infer_share_from_last_pc_stock
            If true, do not use share values from the dataset,
            but rather compute the share using the last known PC stock
        """
        if share and not infer_share_from_last_pc_stock:
            years = self.years_share
            try:
                values = self.values_share * (100 if percent else 1)
            except AttributeError:
                warnings.warn(
                    "You didn't supply share values but asking me to plot them.",
                    UserWarning,
                )
                raise
        else:
            years = self.years
            if share:
                if percent:
                    share_denominator = self.ref_pc_stock / 100
                else:
                    share_denominator = self.ref_pc_stock
            else:
                share_denominator = 1
            values = self.values / share_denominator

        return years, values

    def plot_data(
        self,
        ax=None,
        color="C0",
        share=False,
        percent=False,
        infer_share_from_last_pc_stock=False,
        **kwargs,
    ):
        """
        Plot historic BEC data.

        Parameters
        ----------
        ax
            Matplotlib axes object
        color
            Color of the data points
        share
            TODO correct this: If true, don't plot absolute fleet size, but share of total
            last known value (this is not accurate for the years before!!)
        percent
            If True, plot percent values instead of fraction.
        infer_share_from_last_pc_stock
        """
        if ax is None:
            ax = plt.gca()

        years, values = self.get_data(
            share=share,
            percent=percent,
            infer_share_from_last_pc_stock=infer_share_from_last_pc_stock,
        )
        ax.plot(
            years,
            values,
            **(
                dict(
                    c=color,
                    label=self.plot_label_data,
                )
                | kwargs
            ),
        )

    @property
    def valid_errors(self):
        """
        Check t_0 variance for plausibility
        """
        return self.pcov[-1, -1] < 20

    def get_intersection(self, f, initial_guess):
        """
        Return numerically determined intersection of the
        BEV fleet size with the supplied function `f`.

        Parameters
        ----------
        f
            Function that the intersection is found with
        initial_guess
            Initial guess for t_intersect, used by fsolve

        Returns
        -------
        (t_l, t, t_u)
            Lower confidence bound, expected value, upper confidence bound

        """

        (t,) = scipy.optimize.fsolve(
            lambda x: f(x) - self.ftrans(self.nom(x)), [initial_guess]
        )

        if abs(self.ftrans(self.nom(t)) - f(t)) > 0.5:
            raise RuntimeError(f"Intersection did not converge for nominal value")

        if self.valid_errors:
            try:
                (t_l,) = scipy.optimize.fsolve(
                    lambda x: f(x) - self.ftrans(self.ucb(x)), [initial_guess - 1]
                )
            except:
                t_l = np.nan
            else:
                if abs(self.ftrans(self.ucb(t_l)) - f(t_l)) > 0.5:
                    raise RuntimeError(
                        f"Intersection did not converge for for lower confidence bound"
                    )

            try:
                (t_u,) = scipy.optimize.fsolve(
                    lambda x: f(x) - self.ftrans(self.lcb(x)), [initial_guess + 1]
                )
            except:
                t_u = np.nan
            else:
                if abs(self.ftrans(self.lcb(t_u)) - f(t_u)) > 0.5:
                    raise RuntimeError(
                        f"Intersection did not converge for upper confidence bound"
                    )
        else:
            t_l = t_u = np.nan

        return t_l, t, t_u


# ### Model functions 

# #### Exponential

# +
def f_exp(t, a, t_0, unumpy: bool):
    return a * (t - t_0)


def get_bev_model_exp(
    *, bec_stock, pc_stock_latest, bec_share: pd.Series | None = None
):
    return BEVModel(
        ser=bec_stock,
        ser_share=bec_share,
        model=f_exp,
        ref_pc_stock=pc_stock_latest,
        transformation=(np.exp, np.log),
        alpha=0.05,
        plot_label=model_labels_dict["exp"],
        plot_label_data=label_BEC,
        guess=(0.46, 2010),
        p=2,
        plot_color="C1",
    )


# -

# model test
model = get_bev_model_exp(
    bec_stock=bec_stock_by_region["Germany"],
    pc_stock_latest=pc_stock_by_region_world.loc[2020, "Germany"],
).fit(debug=True)


# #### Logistic

# +
def f_log(t, a, t_0, L, unumpy: bool):
    if unumpy:
        log = unp.log
        pow = unp.pow
        exp = unp.exp
    else:
        log = np.log
        pow = np.power
        exp = np.exp

    return log(L / (1 + exp(-a * (t - t_0))))


def get_bev_model_log(
    *, bec_stock, pc_stock_latest, bec_share: pd.Series | None = None
):
    return BEVModel(
        ser=bec_stock,
        ser_share=bec_share,
        model=ft.partial(f_log, L=pc_stock_latest),
        ref_pc_stock=pc_stock_latest,
        transformation=(np.exp, np.log),
        alpha=0.05,
        guess=(0.1, 2010),
        p=2,
        plot_label=model_labels_dict["log"],
        plot_label_data=label_BEC,
        plot_color="C2",
    )


# model test
model = get_bev_model_log(
    bec_stock=bec_stock_by_region["Germany"],
    pc_stock_latest=pc_stock_by_region_world.loc[2020, "Germany"],
).fit(debug=True)


# -
# #### Bass diffusion

# +
def f_bass(t, p, q, t_0, L, unumpy: bool):
    if unumpy:
        log = unp.log
        pow = unp.pow
        exp = unp.exp
    else:
        log = np.log
        pow = np.power
        exp = np.exp

    return log(
        L * (1 - exp(-(p + q) * (t - t_0))) / (1 + (q / p) * exp(-(p + q) * (t - t_0)))
    )


def get_bev_model_bass(
    *, bec_stock, pc_stock_latest, bec_share: pd.Series | None = None
):
    return BEVModel(
        ser=bec_stock,
        ser_share=bec_share,
        model=ft.partial(f_bass, L=pc_stock_latest),
        ref_pc_stock=pc_stock_latest,
        transformation=(np.exp, np.log),
        alpha=0.05,
        guess=(0.1, 0.1, 1990),
        bounds=((1e-7, 1e-7, 1970), (3, 3, 2030)),
        p=3,
        plot_label=model_labels_dict["bass"],
        plot_label_data=label_BEC,
        plot_color="C3",
    )


# -
# model test
region = "Portugal"
model = get_bev_model_bass(
    bec_stock=bec_stock_by_region[region],
    pc_stock_latest=pc_stock_by_region_europe_us.loc[2019, region],
)
model.fit(debug=True)
# ### Region Handler Class


class RegionHandler:
    """
    Class to handle all regions analyzed.
    Combines and manages data in a sensible way.
    """

    models = {}
    model_definitions = {
        "exp": get_bev_model_exp,
        "log": get_bev_model_log,
        "bass": get_bev_model_bass,
    }

    super_regions = {
        "Europe": [
            "Belgium",
            "Denmark",
            "Finland",
            "France",
            "Germany",
            "Greece",
            "Iceland",
            "Italy",
            "Netherlands",
            "Norway",
            "Poland",
            "Portugal",
            "Spain",
            "Sweden",
            "Switzerland",
            "United Kingdom",
            "Other Europe",
        ],
        "South America": ["Brazil", "Chile", "Mexico"],
        "North America": ["Canada", "USA"],
        "Asia": ["China", "India", "Japan", "Korea"],
        "Australia and Oceania": ["Australia", "New Zealand"],
        "South Africa": ["South Africa"],
        "World": ["World"],
    }

    def __init__(
        self,
        pc_stock: pd.DataFrame,
        bec_stock: pd.DataFrame,
        bec_share: pd.DataFrame | None = None,
    ):
        """
        Initialize RegionHandler class.

        Parameters
        ----------
        pc_stock
            PC stock DataFrame
        bec_stock
            BEC stock DataFrame
        bec_share
            BEC share DataFrame
        """
        self.pc_stock = pc_stock.copy()
        self.bec_stock = bec_stock.copy()
        self.bec_share = bec_share.copy()

        self.all_regions = set(self.bec_stock) | set(self.bec_share)
        for super_region, regions in self.super_regions.items():
            assert set(regions).issubset(
                self.all_regions
            ), f"super_region {super_region} contains regions which we don't have data on"

    @ft.lru_cache
    def get_regions(self, which="regions", subset_bec=False):
        """
        Return regions managed by the RegionHandler.

        Parameters
        ----------
        which
            Which regions to report. Valid values are individual regions, "regions", "super_regions", and the superset "all".
        subset_bec
            Only report regions on which BEC stock data is available

        Returns
        -------
        regions: List[str]
        """
        match which:
            case "regions":
                regions = self.all_regions
            case "super_regions":
                regions = set(self.super_regions)
            case "all":
                regions = self.all_regions | set(self.super_regions)
            case _:
                if which in self.super_regions:
                    regions = set(self.super_regions[which])
                elif which in self.all_regions:
                    regions = {which}
                else:
                    raise ValueError(f"Invalid region selection 'which={which}'")

        if subset_bec:
            regions &= set(self.bec_stock)

        return sorted(regions)

    def expand_regions(self, regions, subset_bec=True):
        """
        Expand regions into elementary regions.

        Parameters
        ----------
        regions
            Regions to expand
        subset_bec
            Constrain expansion on regions on which BEC stock data is available

        Returns
        -------
        expanded_regions: List[str]

        """
        expanded_regions = [
            *it.chain(
                *[
                    (
                        (region,)
                        if region[0] != "*"
                        else self.get_regions(
                            region[1:],
                            subset_bec=subset_bec,
                        )[::-1]
                    )
                    for region in regions
                ]
            )
        ]
        return expanded_regions

    def get_model_types(self):
        """
        Return all available model types

        Returns
        -------
        model_types: List[str]
        """
        return [*self.model_definitions]

    def _populate_super_region_bec_stock(self, region):
        """
        Populate BEC stock of a super region.

        Parameters
        ----------
        region
            The super region to populate the BEC stock of.
        """
        regions = self.get_regions(region, subset_bec=True)
        self.bec_stock[region] = self.bec_stock[regions].ffill(axis=0).sum(axis=1)
        self.bec_stock.loc[self.bec_stock[region] == 0, region] = np.nan

    def _setup_model(self, region, model_type):
        """
        Set up and fit the BEVModel for the specified region and model_type

        Parameters
        ----------
        region
            Region to set up
        model_type
            Model type to set up
        """
        if region not in self.bec_stock:
            self._populate_super_region_bec_stock(region)

        bec_stock = self.bec_stock[region].dropna()
        bec_share = self.bec_share[region].dropna()
        model = self.model_definitions[model_type](
            bec_stock=bec_stock,
            pc_stock_latest=self.get_pc_stock_latest(region)[1],
            bec_share=bec_share,
        )
        try:
            model.fit()
        except Exception as e:
            warnings.warn(f"{model_type} fit failed for {region}, error was {e}")
        else:
            self.models[(region, model_type)] = model

    def get_model(self, region, model_type):
        """
        Return the desired BEVModel.

        Parameters
        ----------
        region
            Region to use
        model_type
            Type of the model to return

        Returns
        -------
        bev_model: BEVModel
        """
        if not (region, model_type) in self.models:
            self._setup_model(region, model_type)

        return self.models[(region, model_type)]

    @ft.lru_cache
    def get_pc_stock_for_year(self, region, year, report_source=False):
        """
        Return the PC stock for the given year.

        Parameters
        ----------
        region
            Region to use
        year
            Year to use
        report_source
            If true, return source in addition to the stock value

        Returns
        -------
        pc_stock: float
        sources: List[str]
            optional
        """
        precedence_order = ["fhwa", "iceland", "oica+eurostat", "oica", "eurostat"]

        regions = self.super_regions.get(region, [region])

        df = pc_stock_by_region.ffill().loc[year, regions].unstack()
        df["agg"] = np.nan
        sources = []
        for source in precedence_order:
            if source in df:
                tmp = df["agg"].fillna(df[source])
                if not df["agg"].equals(tmp):
                    sources.append(source)
                    df["agg"] = tmp

        pc_stock = df["agg"].sum()

        if not report_source:
            return pc_stock
        else:
            return pc_stock, sources

    def get_pc_stock_latest(self, region):
        """
        Returns the latest PC stock, using the year provided.

        Parameters
        ----------
        region
            Region to query the PC stock for
        year
            Reference year to use as the latest known date

        Returns
        -------
        year: int
        pc_stock: float
        """
        return 2020, self.get_pc_stock_for_year(region, year=2020)

    @ft.lru_cache
    def get_intersection_time(
        self, region, model_type, pc_fraction=0.5, initial_guess=None
    ):
        """
        Return the time at which the given PC fraction is reached
        by projected BEC stock.

        Parameters
        ----------
        region
            Region to consider
        model_type
            Model type to use
        pc_fraction
            Desired PC fraction
        initial_guess
            Initial guess for the intersection time

        Returns
        -------
        (t_l, t, t_u): Tuple[float, float, float]
            Lower confidence bound, expected value, upper confidence bound
        """
        if initial_guess is None:
            a, t_0 = rh.get_model(region, "log").popt
            initial_guess = -1 / a * m.log(1 / pc_fraction - 1) + t_0

        model = self.get_model(region, model_type)

        try:
            fraction_abs_fleet = self.get_pc_stock_latest(region)[1] * pc_fraction
            intersection = model.get_intersection(
                lambda x: fraction_abs_fleet,
                initial_guess=initial_guess,
            )
        except:
            print(
                f"You asked me to find the intersection at {pc_fraction=} for {region=} and {model_type=},"
                f" which failed. Initial guess was {initial_guess:.3f} at {fraction_abs_fleet} cars. ftrans values are:"
                f"\n- lower bound: {model.ftrans(model.ucb(initial_guess))}"
                f"\n- nominal: {model.ftrans(model.nom(initial_guess))}"
                f"\n- upper bound: {model.ftrans(model.lcb(initial_guess))}"
            )
            raise

        return intersection

    def get_pc_stock_series(self, region: str, source="oica"):
        """
        Return PC stock series

        Parameters
        ----------
        region
            Region to return the data for
        source
            Data source to use

        Returns
        -------
        pc_stock: pd.Series
        """
        return self.pc_stock.loc[:, (region, source)]


rh = RegionHandler(
    bec_stock=bec_stock_by_region,
    pc_stock=pc_stock_by_region,
    bec_share=bec_share_by_region,
)

regions_europe = rh.super_regions["Europe"]
rois = ["World", "Europe"] + regions_europe + ["USA"]


# ## Construct and Render Plots

# ### Figure 1: Worldwide Exponential Adoption

# #### Setup

class OOMFormatter(mpl.ticker.ScalarFormatter):
    """
    Order Of Magnitude formatter.

    Stolen from: https://stackoverflow.com/a/42658124
    """

    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        mpl.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText
        )

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format


# +
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_world_adoption(ax):
    """
    Create a plot of the global BEC adoption in linear scale,
    with an additional logarithmically-scaled inset.
    """
    model_bev_exp_world = deepcopy(rh.get_model("World", "exp"))
    model_bev_exp_world.fit(year_range=(2016, 2022))

    axins = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.0, 0.5, 0.5, 0.5),
        bbox_transform=ax.transAxes,
        loc=2,
    )

    ax.plot(
        bec_stock_by_region.index,
        bec_stock_by_region["World"],
        markersize=8,
        marker=".",
        lw=2,
    )

    ax.yaxis.set_major_formatter(OOMFormatter(6, "%1.f"))
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    ax.set_xlabel(label_date)
    ax.set_ylabel(f"Worldwide {label_BEC_stock}")

    axins.plot(
        bec_stock_by_region.index,
        bec_stock_by_region["World"],
        markersize=8,
        marker=".",
        lw=2,
    )

    axins.plot(
        *evf(
            np.linspace(2017, 2022, 200),
            lambda x: model_bev_exp_world.ftrans(model_bev_exp_world.nom(x) - 1),
        ),
        c="k",
        alpha=1,
        dashes=(1, 0.5),
        lw=1.5,
    )

    axins.tick_params(
        left=False,
        right=True,
        labelleft=False,
        labelright=True,
        which="both",
    )
    axins.xaxis.set_major_locator(mpl.ticker.MultipleLocator(3))
    axins.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    axins.set_yscale("log")


# -
# #### Render figure

# +
fig = plt.figure(
    figsize=(fig_width, min(fig_width / 3 * 2, fig_height_max)),
    dpi=150,
)

gs = mpl.gridspec.GridSpec(
    1,
    1,
    figure=fig,
    hspace=0,
    wspace=0,
    bottom=0.2,
    top=0.88,
    left=0.20,
    right=0.86,
)

ax = fig.add_subplot(gs[0])

plot_world_adoption(ax)

fpath = plot_dir / "worldwide exponential bec adoption.pdf".replace(" ", "-")
fig.savefig(fpath, dpi=fig_dpi)
alt_fpath = fpath.parent / "figure-1.pdf"
if not alt_fpath.exists():
    alt_fpath.symlink_to(fpath)


# -
# ### Figure 2: Historic Trajectory Comparison across Regions

# #### Setup

def plot_models_region_comparison_super_regions(
    ax,
    super_regions,
    share=True,
    percent=True,
    infer_share_from_last_pc_stock=False,
    model_type="log",
    super_region_colors=None,
    xlim=None,
    plot_models=True,
    plot_data=False,
    annotate: list | None = None,
    annotate_offsets=None,
    annotate_fractions=None,
    annotate_years=None,
    annotate_mode="model",
):
    """
    Parameters
    ----------
    ax
        Matplotlib axes to use
    super_regions
        Super region to plot
    share
        If True, plot share instead of absolute values
    percent
        If True and plotting share, show percentages
    infer_share_from_last_pc_stock
        If True, infer BEC share from last known PC stock value
    model_type
        BEC Model type to use
    super_region_colors
        Colors for super region curves
    xlim
        Time range to plot
    plot_models
        If True, plot models
    plot_data
        If True, plot historic data
    annotate
        Optional, annonate the curves, ordered by adoption speed
        (by log intersection date).
        Example: [0, 5, -1]. May also be 'm' for median.
    annotate_offsets
        List of XY offsets for annotation labels
    annotate_fractions
        Anchor arrows at specified modeled PC fraction values.
        Takes precedence over annotate_years
    annotate_years
        Anchor arrows at specified years (useful for plotting
        data at small adoption values). If supplied in addition to `annotate_fractions`,
        the latter takes precendence over `annotate_years`.
    annotate_mode
        Base for the annotation specifications, may be 'model' or 'data'.
    """
    if xlim is None:
        xlim = 2020, 2040

    if super_region_colors is None:
        super_region_colors = [f"C{i}" for i in range(len(super_regions))]

    for super_region, super_region_color in zip(super_regions, super_region_colors):
        if plot_models:
            rh.get_model(super_region, model_type).plot(
                ax=ax,
                label=super_region,
                c=super_region_color,
                lw=4,
                alpha=0.8,
                share=share,
                percent=percent,
                zorder=100,
                t_min_max=xlim,
            )
        if plot_data:
            rh.get_model(super_region, model_type).plot_data(
                ax=ax,
                label=super_region if not plot_models else "_",
                c=super_region_color,
                ls="-",
                alpha=0.8,
                marker="o",
                lw=4,
                share=share,
                percent=percent,
                infer_share_from_last_pc_stock=infer_share_from_last_pc_stock,
                zorder=100,
            )

        regions = rh.expand_regions([f"*{super_region}"])
        for region in regions:
            if plot_models:
                rh.get_model(region, model_type).plot(
                    ax=ax,
                    label="_",
                    c=super_region_color,
                    ls="-",
                    alpha=0.5,
                    share=share,
                    percent=percent,
                    t_min_max=xlim,
                )
            if plot_data:
                rh.get_model(region, model_type).plot_data(
                    ax=ax,
                    label="_",
                    c=super_region_color,
                    ls="-",
                    marker=".",
                    alpha=0.5,
                    share=share,
                    percent=percent,
                    infer_share_from_last_pc_stock=infer_share_from_last_pc_stock,
                )

    if annotate is not None:
        # TODO this currently only works properly for the model curves, not for data
        if annotate_offsets is None:
            annotate_offsets = [[0, 0]] * len(annotate)
        elif (n_missing_offsets := len(annotate) - len(annotate_offsets)) > 0:
            annotate_offsets += [[0, 0]] * n_missing_offsets

        if annotate_fractions is not None and annotate_years is not None:
            # can't use both
            warnings.warn(
                "Caution: I got both `annotate_fractions` and `annotate_years`. The former takes precedence (!)",
                UserWarning,
            )
            annotate_years = None
        elif annotate_fractions is None and annotate_years is None:
            # generate some defaults
            annotate_fractions = [0.5] * len(annotate)
        elif (
            annotate_fractions is not None
            and (n_missing_fractions := len(annotate) - len(annotate_fractions)) > 0
        ):
            annotate_fractions += [0.5] * n_missing_fractions

        regions = [
            *it.chain.from_iterable(
                rh.expand_regions([f"*{super_region}"])
                for super_region in super_regions
            )
        ]

        inters_times = pd.Series(
            [
                rh.get_intersection_time(region, model_type=model_type)[1]
                for region in regions
            ],
            index=regions,
        ).sort_values()

        for i, (region, offset, annotate_fraction_or_year) in enumerate(
            zip(
                [
                    inters_times.index[i]
                    if i != "m"
                    else inters_times.index[len(inters_times) // 2]
                    for i in annotate
                ],
                annotate_offsets,
                annotate_fractions if annotate_years is None else annotate_years,
            )
        ):
            if not annotate_years:
                t = rh.get_intersection_time(
                    region, model_type=model_type, pc_fraction=annotate_fraction_or_year
                )[1]
            else:
                t = annotate_fraction_or_year

            model = rh.get_model(region=region, model_type=model_type)

            match annotate_mode:
                case "model":
                    if share:
                        if percent:
                            share_denominator = model.ref_pc_stock / 100
                        else:
                            share_denominator = model.ref_pc_stock
                    else:
                        share_denominator = 1
                    y = model.ftrans(model.nom(t)) / share_denominator
                case "data":
                    years, values = model.get_data(
                        share=share,
                        percent=percent,
                        infer_share_from_last_pc_stock=infer_share_from_last_pc_stock,
                    )
                    y = scipy.interpolate.interp1d(years, values, kind="linear")(t)
                case _:
                    raise ValueError(f"Invalid {annotate_mode=}")

            ax.annotate(
                region,
                xy=(t, y),
                xytext=(
                    t + offset[0],
                    y + offset[1],
                ),
                xycoords="data",
                textcoords="data",
                va="center",
                ha="center",
                arrowprops=dict(
                    arrowstyle="->",
                    color="0.5",
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle=f"arc3,rad=-0.2",
                ),
                bbox=dict(boxstyle="round", linestyle="None", color=(1, 1, 1, 0.7)),
                zorder=100,
            )

    ax.set_xlabel(label_date)
    if share:
        if percent:
            ax.set_ylabel(label_BEC_share_percent)
        else:
            ax.set_ylabel(label_BEC_share)
    else:
        ax.set_ylabel(label_BEC_stock)

    ax.set_xlim(xlim)
    ax.legend(loc=2)


# #### Render figure

# +
fig = plt.figure(
    figsize=(fig_width, min(fig_width / 3 * 2, fig_height_max)),
    dpi=150,
)

gs = mpl.gridspec.GridSpec(
    1,
    1,
    figure=fig,
    hspace=0,
    wspace=0,
    bottom=0.2,
    top=0.88,
    left=0.20,
    right=0.86,
)

ax = fig.add_subplot(gs[0])

plot_models_region_comparison_super_regions(
    ax=ax,
    super_regions=["World", "Europe", "USA"],
    model_type="log",
    xlim=(2011, 2023),
    plot_data=True,
    plot_models=False,
    share=True,
    percent=True,
    infer_share_from_last_pc_stock=False,
    annotate=[0, 1, -3],
    annotate_offsets=[[-2, 0.5], [-2, 0], [-1.8, 1.6]],
    annotate_years=[2014.5, 2018.5, 2021.7],
    annotate_mode="data",
)

ax.set_xlim(2011, 2023)
ax.set_ylim(-0.5, 6.5)

ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

fpath = plot_dir / "adoption fraction curves data.pdf".replace(" ", "-")
fig.savefig(fpath, dpi=fig_dpi)
alt_fpath = fpath.parent / "figure-2.pdf"
if not alt_fpath.exists():
    alt_fpath.symlink_to(fpath)


# -
# ### Figure 3: Dominance Time Estimation Construction

# #### Setup

def plot_intersection_construction(
    region,
    *,
    ax,
    model_colors_dict=None,
    model_labels_dict=None,
    bec_data_color: str = "b",
    pc_data_color: str = "k",
    pc_source: str = "oica",
):
    """
    Create a plot of the intersection construction used to find the
    estimated dominance time.

    Parameters
    ----------
    ax
        Matplotlib axes to use
    model_colors_dict
        Color mapping for the BEC models
    model_labels_dict
        Label mapping for the BEC models
    bec_data_color
        Color of the BEC data points
    pc_data_color
        Color of the PC data points
    pc_source
        Source for PC data points
    """
    model_colors_dict = model_colors_dict or {}

    t_min, t_max = 2010.5, 2035.5

    model_types = rh.get_model_types()
    model_colors_dict = model_colors_dict or {}
    model_labels_dict = model_labels_dict or {}

    model_colors = {
        model_type: model_colors_dict.get(model_type, f"C{i}")
        for i, model_type in enumerate(model_types, 1)
    }
    model_labels = {
        model_type: model_labels_dict.get(model_type, model_type)
        for i, model_type in enumerate(model_types, 1)
    }
    for model_type in model_types:
        model = rh.get_model(region=region, model_type=model_type)
        color = model_colors[model_type]

        model.plot(c=color, label=model_labels[model_type], t_min_max=(2011, t_max))
        ax.axvline(
            rh.get_intersection_time(region, model_type)[1],
            alpha=0.5,
            ls="-",
            c=color,
        )

    model.plot_data(
        **{"markersize": 8, "marker": ".", "linestyle": "none"},
        color=bec_data_color,
        label="BEC",
    )

    pc_stock = rh.get_pc_stock_series(region, pc_source)
    ax.plot(
        pc_stock.index,
        pc_stock,
        color=pc_data_color,
        linestyle="none",
        marker="s",
        markersize=6,
        label="Total PC",
    )

    pc_stock_latest = rh.get_pc_stock_latest(region)[1]
    print(pc_stock.dropna().iloc[-1], pc_stock_latest)
    ax.axhline(pc_stock_latest, c="k", zorder=0)
    ax.axhline(pc_stock_latest / 2, dashes=(1, 1), c="k", zorder=0, lw=1)
    ax.set_yscale("log")
    ax.set_ylim(100, 50e7)

    maj_loc = mpl.ticker.LogLocator(subs=(1,), numticks=10)
    min_loc = mpl.ticker.LogLocator(subs=np.r_[0.2:1:0.1], numticks=10)
    ax.yaxis.set_major_locator(maj_loc)
    ax.yaxis.set_minor_locator(min_loc)
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

    ax.set_xlabel(label_date)
    ax.set_ylabel(f"Fleet size in {region}")

    ax.set_xlim(t_min, t_max)

    ax.legend(loc=4)


# #### Render figure

# +
fig = plt.figure(
    figsize=(fig_width, min(fig_width / 3 * 2, fig_height_max)),
    dpi=150,
)

gs = mpl.gridspec.GridSpec(
    1,
    1,
    figure=fig,
    hspace=0,
    wspace=0,
    bottom=0.2,
    top=0.88,
    left=0.20,
    right=0.86,
)

ax = fig.add_subplot(gs[0])

plot_intersection_construction(
    "Germany",
    ax=ax,
    model_labels_dict=model_labels_dict,
    model_colors_dict=model_colors_dict,
    bec_data_color="k",
    pc_source="eurostat",
)

fpath = plot_dir / "intersection construction germany.pdf".replace(" ", "-")
fig.savefig(fpath, dpi=fig_dpi)
alt_fpath = fpath.parent / "figure-3.pdf"
if not alt_fpath.exists():
    alt_fpath.symlink_to(fpath)
# -

# ### Figure 4: Model Adoption Trajectory Comparison across Regions

# #### Render figure

# +
fig = plt.figure(
    figsize=(fig_width, min(fig_width / 3 * 2, fig_height_max)),
    dpi=150,
)


gs = mpl.gridspec.GridSpec(
    1,
    2,
    figure=fig,
    width_ratios=[1, 1],
    hspace=0.4,
    wspace=0.05,
    bottom=0.2,
    top=0.8,
    right=0.95,
)

axs = [fig.add_subplot(gs) for gs in gs]

panel_1_label = axs[0].text(
    0.10, 1.05, r"\textbf{(a)}", transform=axs[0].transAxes, ha="right", va="bottom"
)

panel_2_label = axs[1].text(
    0.10, 1.05, r"\textbf{(b)}", transform=axs[1].transAxes, ha="right", va="bottom"
)
plot_models_region_comparison_super_regions(
    ax=axs[0],
    super_regions=["World", "Europe", "USA"],
    model_type="log",
    xlim=(2023, 2035),
    annotate=[
        0,
        1,
        -2,
        -1,
    ],
    annotate_offsets=[
        [3, -7],
        [3.5, -2],
        [-3, 8],
        [2.2, -4],
    ],
    annotate_fractions=[
        0.35,
        0.16,
        0.30,
        0.1,
    ],
)

plot_models_region_comparison_super_regions(
    ax=axs[1],
    super_regions=["World", "Europe", "USA"],
    model_type="bass",
    xlim=(2023, 2035),
    annotate=[
        0,
        1,
        -2,
        -1,
    ],
    annotate_offsets=[
        [3, -8],
        [-3, 5],
        [-3, 5],
        [-3, 10],
    ],
    annotate_fractions=[
        0.37,
        0.66,
        0.3,
        0.06,
    ],
)

axs[0].set_ylim(-3, 90)
axs[0].axhline(50, dashes=(1, 1), c="k", zorder=-100, lw=1)

axs[1].set_ylim(-3, 90)
axs[1].axhline(50, dashes=(1, 1), c="k", zorder=-100, lw=1)

axs[0].get_legend().set_zorder(300)

axs[1].get_legend().remove()
axs[1].yaxis.label.set_visible(False)
axs[1].yaxis.set_tick_params(left=False, labelleft=False)

for ax in axs:
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4, integer=True))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(10, integer=True))

axs[0].set_xticks(ticks=axs[0].get_xticks()[:-1], labels=axs[0].get_xticklabels()[:-1])
axs[1].set_xticks(ticks=axs[1].get_xticks()[1:], labels=axs[1].get_xticklabels()[1:])

axs[0].set_title(model_labels_dict["log"])
axs[1].set_title(model_labels_dict["bass"])

fpath = plot_dir / "adoption fraction curves log bass.pdf".replace(" ", "-")
fig.savefig(fpath, dpi=fig_dpi)
alt_fpath = fpath.parent / "figure-4.pdf"
if not alt_fpath.exists():
    alt_fpath.symlink_to(fpath)


# -


# ### Figure 5: Dominance Estimate Comparison Chart

# #### Setup

def plot_dominance_estimate_comparison_chart(
    ax,
    regions,
    summary_regions=None,
    summary_model_types=None,
    summary_region_label_offsets=None,
    summary_labels=False,
    summary_region_colors=None,
    model_colors_dict=None,
):
    """
    Plot comparison chart of all the dominance estimates.
    Shows all three models for individual regions,
    and specific model types for additional summary
    regions as single vertical lines.

    Parameters
    ----------
    regions
        Regions to show
    summary_regions
        Summary regions to show
    summary_model_types
        Model types to show as vertical lines for summary regions
    summary_region_label_offsets
        If labeling summary regions, use these offset values
    summary_labels
        Optional labels for summary regions
    summary_region_colors
        Regions for vertical summary region lines
    model_colors_dict
        Color mapping for the BEC models
    """
    if regions is None:
        regions = []

    if summary_regions is None:
        summary_regions = []

    if summary_model_types is None:
        summary_model_types = []

    model_colors_dict = model_colors_dict or {}

    regions = rh.expand_regions(regions)
    summary_regions = rh.expand_regions(summary_regions)

    if summary_region_colors is None:
        summary_region_colors = ["C0"] * len(summary_regions)

    if summary_region_label_offsets is None:
        summary_region_label_offsets = [[0, 0]] * len(summary_regions)
    model_types = rh.get_model_types()
    model_colors = {
        model_type: model_colors_dict.get(model_type, f"C{i}")
        for i, model_type in enumerate(model_types, 1)
    }

    time_point_plot_kwargs = dict(
        marker="|",
        alpha=1,
        ms=4,
        markeredgewidth=3,
        zorder=100,
        ls="none",
    )
    inters_times = pd.DataFrame(
        {
            model_type: {
                region: rh.get_intersection_time(region, model_type)[1]
                for region in regions
            }
            for model_type in model_types
        }
    )
    inters_times.sort_values("log", inplace=True, ascending=False)
    inters_times.index.rename("region", inplace=True)
    inters_times.reset_index(drop=False, inplace=True)

    for model_type, ser in inters_times.iloc[:, 1:].items():
        ax.plot(
            ser.values,
            inters_times.index,
            color=model_colors[model_type],
            **time_point_plot_kwargs,
        )

    for i, row in inters_times.iterrows():
        min_estimate = row[model_types].min()
        max_estimate = row[model_types].max()

        ax.plot(
            [min_estimate, max_estimate],
            [i] * 2,
            "k-",
            lw=1,
            alpha=0.8,
            zorder=10,
        )

    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)

    summary_linestyles = ["-", (0.5, 0.5)]
    for summary_linestyle, summary_model_type in zip(
        summary_linestyles, summary_model_types
    ):
        for summary_region, summary_region_label_offset, summary_region_color in zip(
            summary_regions,
            summary_region_label_offsets,
            summary_region_colors,
        ):
            time = rh.get_intersection_time(
                region=summary_region, model_type=summary_model_type
            )[1]
            kwargs = {}
            if isinstance(summary_linestyle, str):
                kwargs["linestyle"] = summary_linestyle
            else:
                kwargs["dashes"] = summary_linestyle
            ax.axvline(
                time,
                label=f"{'_' if summary_linestyle != '-' else ''}{summary_region}",
                color=summary_region_color,
                alpha=0.8,
                **kwargs,
            )
            if summary_labels:
                ax.annotate(
                    summary_region,
                    xy=(time, 1),
                    xytext=(
                        time + summary_region_label_offset[0],
                        1 + summary_region_label_offset[1],
                    ),
                    xycoords=trans,
                    textcoords=trans,
                    va="bottom",
                    ha="center",
                    transform=trans,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="0.5",
                        shrinkA=12,
                        shrinkB=5,
                        patchA=None,
                        patchB=None,
                        connectionstyle="arc3,rad=0.0",
                    ),
                )

    ax.set_yticks(ticks=inters_times.index, labels=inters_times.region)

    ax.set_xticks(np.r_[2022:2041:3])
    ax.set_xticks(np.r_[2022:2041:1], minor=True)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", which="minor", lw=0.3, zorder=0)
    ax.grid(True, axis="x", which="major", lw=0.9, zorder=0)

    ax.grid(True, axis="y", which="major", lw=0.3, zorder=0)
    ax.set_xlabel("year")

    handles, labels = ax.get_legend_handles_labels()
    legend_dict = {
        model_labels_dict["exp"]: mpl.lines.Line2D(
            [0], [0], color=model_colors["exp"], **time_point_plot_kwargs
        ),
        model_labels_dict["log"]: mpl.lines.Line2D(
            [0], [0], color=model_colors["log"], **time_point_plot_kwargs
        ),
        model_labels_dict["bass"]: mpl.lines.Line2D(
            [0], [0], color=model_colors["bass"], **time_point_plot_kwargs
        ),
    }

    ax.legend(
        handles=[*legend_dict.values()] + handles,
        labels=[*legend_dict.keys()] + labels,
        loc=1,
        bbox_to_anchor=(1.63, 1),
    )


# #### Render figure

# +
fig = plt.figure(
    figsize=(fig_width, min(fig_width * 1 / 1.4, fig_height_max)),
    dpi=150,
)

gs = mpl.gridspec.GridSpec(
    1,
    1,
    figure=fig,
    hspace=0,
    wspace=0,
    bottom=0.18,
    top=0.90,
    left=0.25,
    right=0.70,
)

ax = fig.add_subplot(gs[0])

plot_dominance_estimate_comparison_chart(
    ax=ax,
    regions=["*Europe"],
    summary_regions=["World", "Europe", "USA"],
    summary_model_types=["log", "bass"],
    summary_region_colors=["C0", "C1", "C2"],
    model_colors_dict=model_colors_dict,
)

fpath = plot_dir / "comparison of exp log bass models for europe.pdf".replace(" ", "-")
fig.savefig(fpath, dpi=fig_dpi)
alt_fpath = fpath.parent / "figure-5.pdf"
if not alt_fpath.exists():
    alt_fpath.symlink_to(fpath)


# -
# ### Figure 6: Distributions of BEC fractions and growth rates

# #### Setup

def plot_bec_fraction_hist(ax=ax):
    ax.hist(
        bec_share_by_region.loc[2022, [*(set(rois) - {"World", "Europe"})]]
        .dropna()
        .values
        * 100,
        bins=np.logspace(np.log10(0.1), np.log10(50.0), 11),
        rwidth=0.9,
    )


def plot_growth_rate_hist(ax=ax):
    a_values = np.array([rh.get_model(roi, "exp").popt[0] for roi in rois])

    a_values_wo_agg = np.array(
        [rh.get_model(roi, "exp").popt[0] for roi in set(rois) - {"World", "Europe"}]
    )

    ax.hist(
        a_values_wo_agg,
        bins=np.r_[0:1.0:11j],
        rwidth=0.9,
        zorder=0,
    )

    mean = a_values_wo_agg.mean()
    std = a_values_wo_agg.std()
    color = "k"

    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)

    line_mean = ax.axvline(mean, lw=1, ymax=1.02, zorder=10, color=color)
    line_minus_sigma = ax.axvline(
        mean - std, lw=1, ymax=1.02, dashes=(1, 1), zorder=10, color=color
    )
    line_plus_sigma = ax.axvline(
        mean + std, lw=1, ymax=1.02, dashes=(1, 1), zorder=10, color=color
    )

    line_mean.set_clip_on(False)
    line_minus_sigma.set_clip_on(False)
    line_plus_sigma.set_clip_on(False)
    print(f"{mean:.2f} +- {std:.2f}")

    ax.text(
        mean,
        1.03,
        r"$\overline a$",
        transform=trans,
        ha="center",
        va="bottom",
    )


# #### Render figure

# +
fig = plt.figure(
    figsize=(fig_width, min(fig_width / 3 * 2, fig_height_max)),
    dpi=150,
)


gs = mpl.gridspec.GridSpec(
    1,
    2,
    figure=fig,
    width_ratios=[1, 1],
    hspace=0.4,
    wspace=0.05,
    bottom=0.2,
    top=0.8,
    right=0.95,
)

axs = [fig.add_subplot(gs) for gs in gs]

panel_1_label = axs[0].text(
    0.10, 1.05, r"\textbf{(a)}", transform=axs[0].transAxes, ha="right", va="bottom"
)

panel_2_label = axs[1].text(
    0.10, 1.05, r"\textbf{(b)}", transform=axs[1].transAxes, ha="right", va="bottom"
)

plot_bec_fraction_hist(ax=axs[0])
plot_growth_rate_hist(ax=axs[1])

axs[1].yaxis.label.set_visible(False)
axs[1].yaxis.set_tick_params(left=False, labelleft=False)

axs[0].set_xscale("log")
axs[0].set_xlim(0.094, 56)
axs[0].set_xlabel(r"BEC fraction $[\%]$")
axs[0].set_ylabel(r"Number of countries")

axs[1].set_xlim(0, 1)
axs[1].set_xlabel(r"Exponential growth rate $a$")
axs[1].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
axs[1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))

ylim = (0, 10.5)
axs[0].set_ylim(*ylim)
axs[1].set_ylim(*ylim)

xticks = [0.1, 1, 10, 30]
axs[0].set_xticks(ticks=xticks, labels=[f"{x:g}" for x in xticks])

fpath = plot_dir / "bec fraction and growth rate distributions.pdf".replace(" ", "-")
fig.savefig(fpath, dpi=fig_dpi)
alt_fpath = fpath.parent / "figure-6.pdf"
if not alt_fpath.exists():
    alt_fpath.symlink_to(fpath)
# -
# ## Convert figures to EPS format

for pdf_path in tqdm(sorted(plot_dir.glob("figure-*.pdf"))):
    pure_path = Path(f"{pdf_path.parent/pdf_path.stem}")
    ps_path = Path(f"{pdf_path.parent/pdf_path.stem}.ps")
    eps_path = Path(f"{pdf_path.parent/pdf_path.stem}.eps")

    ps_path.unlink(missing_ok=True)
    eps_path.unlink(missing_ok=True)
    subprocess.run(
        ["pdf2ps", pdf_path, ps_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ps2eps", ps_path, pure_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ps_path.unlink(missing_ok=True)


# ## Create tables

# ### Table 1: Data sources

# #### Setup

def create_table_data_sources(return_df=False):
    """
    Creates an overview LaTeX table of the data sources

    Parameters
    ----------
    return_df
        If True, additionally return the table as a DataFrame

    Returns
    -------
    out_str: str
        LaTeX table string
    out_df: pd.DataFrame
        Optional DataFrame representation of the table
    """
    has_value_df = ~rh.bec_stock.loc[:, rois].isna()
    bec_data_sources = pd.concat(
        (
            has_value_df.idxmax(axis=0).rename("IEA BEC data start"),
            has_value_df.iloc[::-1].idxmax(axis=0).rename("IEA BEC data end"),
        ),
        axis=1,
    ).sort_index()

    pc_data_sources = pd.Series(
        {
            region: ", ".join(
                rh.get_pc_stock_for_year(region, 2020, report_source=True)[1]
            )
            for region in rois
        }
    )

    l = {
        "oica": "OICA",
        "iceland": "Statistics Iceland",
        "fhwa": "FHWA",
    }
    pattern = re.compile("|".join(l))
    pc_data_sources = pd.DataFrame(
        pc_data_sources.str.replace(pattern, lambda m: l.get(m.group(0)), regex=True)
        .str.replace(r"OICA\+eurostat(, OICA)?", "Eurostat, OICA", regex=True)
        .rename("Data source PC stock")
    )

    out_df = pd.concat(
        [
            pc_data_sources,
            pd.Series(
                "IEA", index=bec_data_sources.index, name="Data source BEC stock"
            ),
            bec_data_sources.agg(
                "[{0[IEA BEC data start]:d}, {0[IEA BEC data end]:d}]".format, axis=1
            ).rename("BEC data range [a]"),
        ],
        axis=1,
    ).loc[rois]

    styler = out_df.style.map_index(
        lambda v: "font-weight: bold;", axis="columns"
    ).format("{:}")

    out_str = styler.to_latex(hrules=True, multicol_align="|c|", convert_css=True)
    out_str = re.sub(
        r"(\\begin{tabular}{)([lr]+)(})",
        lambda m: f"{m[1]}{'|'.join(m[2])}|{m[3]}",
        out_str,
    )
    if return_df:
        return out_str, out_df
    else:
        return out_str


# #### Save table

table_str = create_table_data_sources()
fpath = tables_dir / "data-sources.tex"
fpath.write_text(table_str)
alt_fpath = fpath.parent / "table-1.tex"
if not alt_fpath.exists():
    alt_fpath.symlink_to(fpath)


# ### Table 2: Model coefficients and fit properties

# #### Setup

def create_table_model_coefficients_properties(regions):
    """
    Create LaTeX table of all models' parameters and
    properties for the given regions.

    Parameters
    ----------
    regions: List[str]
        List of regions to include in the table

    Returns
    -------
    out_str
        LaTeX table string
    """
    out_df_dict = defaultdict(dict)
    for region in regions:
        region_dict = {}
        for model_type in rh.get_model_types():
            model = rh.get_model(region, model_type)
            fun = model.model

            fit_param_names = inspect.getfullargspec(fun).args[1:]
            fit_param_values = model.popt
            fit_param_dict = dict(zip(fit_param_names, fit_param_values))

            fixed_param_dict = inspect.getfullargspec(fun).kwonlydefaults or {}

            for param_name, param_value in fit_param_dict.items():
                region_dict[(model_type, "fit_params", param_name)] = param_value

            for param_name, param_value in fixed_param_dict.items():
                region_dict[(model_type, "fixed_params", param_name)] = param_value

            region_dict[
                (model_type, "fit_properties", r"$R^2_\mathrm{adj}$")
            ] = model.r2adj
            region_dict[(model_type, "fit_properties", r"\mathrm{RMSD}")] = model.rmsd

        out_df_dict[region] = region_dict

    out_df = pd.DataFrame(out_df_dict).T  # .sort_index()
    out_df.rename_axis("country", axis=0, inplace=True)
    out_df.rename_axis(["model", "param_type", "param_name"], axis=1, inplace=True)

    out_df.rename(
        model_labels_dict,
        axis=1,
        level="model",
        inplace=True,
    )

    out_df.rename(
        {"fit_params": "Fit", "fixed_params": "Fixed", "fit_properties": "Properties"},
        axis=1,
        level="param_type",
        inplace=True,
    )

    out_df.rename(
        {pn: f"${pn}$" for pn in out_df.columns.levels[-1]},
        axis=1,
        level="param_name",
        inplace=True,
    )

    out_df.rename_axis(["Model", "Parameter type", "Parameter"], axis=1, inplace=True)

    format_dict = {}

    for col in out_df.columns:
        if "RMSD" in col[2]:
            format_dict[col] = "${:.0f}$"
        elif "R^2" in col[2]:
            format_dict[col] = "${:.2f}$"
        elif "m" in col[2]:
            format_dict[col] = "${:.0f}$"
        elif "L" in col[2]:
            format_dict[col] = "${:.0f}$"
        elif "p" in col[2]:
            format_dict[col] = lambda x: f"${scinum(x)}$"
        else:
            format_dict[col] = "${:.2f}$"

    styler = out_df.style.map_index(
        lambda v: "font-weight: bold;", axis="columns"
    ).format(format_dict)

    def fix_tex(s):
        match s:
            case "$x0$":
                return "$x_0$"
            case "$m_$":
                return "$m$"
            case _:
                return s.replace("$$", "$")

    out_df.columns = out_df.columns.set_levels(
        out_df.columns.levels[2].map(fix_tex), level=2
    )

    out_str = styler.to_latex(hrules=True, multicol_align="|c|", convert_css=True)

    out_str = re.sub(
        r"(\\begin{tabular}{)([lr]+)(})",
        lambda m: f"{m[1]}{'|'.join(m[2].replace('r', 'l'))}|{m[3]}",
        out_str,
    )
    out_str = out_str.replace(r"& Fixed &", r"& \multicolumn{1}{|c|}{Fixed} &")
    out_str = re.sub(r"(?m)^country.*\n", "", out_str)

    return out_str


# #### Save table

table_str = create_table_model_coefficients_properties(rois)
fpath = tables_dir / "model-coefficients-properties.tex"
fpath.write_text(table_str)
alt_fpath = fpath.parent / "table-2.tex"
if not alt_fpath.exists():
    alt_fpath.symlink_to(fpath)


# ### Table 3: Transition times $t_{1/2}$

# #### Setup

def create_table_transition_times(regions, return_df=False):
    """
    Creates an overview LaTeX table of dominance time estimates.

    Parameters
    ----------
    regions
        Regions to list
    return_df
        If True, additionally return the table as a DataFrame

    Returns
    -------
    out_str: str
        LaTeX table string
    out_df: pd.DataFrame
        Optional DataFrame representation of the table
    """
    res_dict = {}
    for region in regions:
        res_dict[region] = {}
        for model_type in rh.get_model_types():
            lcb, nom, ucb = rh.get_intersection_time(region, model_type)
            model_header = model_labels_dict[model_type]
            res_dict[region][
                (
                    model_header,
                    rf"$\thalf^{{(\mathrm{{{model_type}}})}}\,[\mathrm{{a}}]$",
                )
            ] = f"${nom:.0f}$"

            if any(np.isnan([lcb, ucb])):
                ci_str = r"--"
            else:
                ci_str = f"$[{m.floor(lcb):.0f},{m.ceil(ucb):.0f}]$"

            res_dict[region][
                (
                    model_header,
                    "CI-95 $[a]$",
                )
            ] = ci_str

    out_df = pd.DataFrame.from_dict(res_dict, orient="index")

    styler = out_df.style.map_index(
        lambda v: "font-weight: bold;", axis="columns"
    ).format("{:}")

    out_str = styler.to_latex(hrules=True, multicol_align="|c|", convert_css=True)
    out_str = re.sub(
        r"(\\begin{tabular}{)([lr]+)(})",
        lambda m: f"{m[1]}{'|'.join(m[2])}|{m[3]}",
        out_str,
    )
    if return_df:
        return out_str, out_df
    else:
        return out_str


# #### Save table

table_str, df = create_table_transition_times(rois, return_df=True)
fpath = tables_dir / "transition-times.tex"
fpath.write_text(table_str)
alt_fpath = fpath.parent / "table-3.tex"
if not alt_fpath.exists():
    alt_fpath.symlink_to(fpath)

assert all(df.iloc[:, [0, 2, 4]].apply(lambda x: x.is_monotonic_increasing, axis=1))

# ## Final considerations

# #### Mean annual PC stock growth

df = rh.pc_stock.loc[
    [2015, 2020], ([*set(rois) - {"Europe", "World"} | {"China", "India"}], "oica")
].T.droplevel("source")

mean_annual_growth = ((df[2020] / df[2015]) ** (1 / (2020 - 2015)) - 1) * 100

mean_annual_growth.sort_values()

mean_annual_growth.describe()



# ### Historic peak sales values comparison

# Peak values according to [CEIC Data](https://www.ceicdata.com/en/indicator/united-states/motor-vehicle-sales-passenger-cars) for the USA and [GoodCarBadCar](https://www.goodcarbadcar.net/greece-car-sales-data/) for all other regions.

peak_values_by_region = pd.Series(
    {
        "Belgium": 636e3,
        "Denmark": 260e3,
        "Finland": 148e3,
        "France": 2.3e6,
        "Germany": 4.2e6,
        "Greece": 290e3,
        "Iceland": 20e3,
        "Italy": 2.5e6,
        "Netherlands": 611e3,
        "Norway": 176e3,
        "Poland": 556e3,
        "Portugal": 277e3,
        "Spain": 1.6e6,
        "Sweden": 435e3,
        "Switzerland": 380e3,
        "United Kingdom": 2.7e6,
        "USA": 7_761_592,
    }
)


def get_sales_max(region, model_type):
    try:
        model = rh.get_model(region=region, model_type=model_type)
        year_range = np.r_[2022 : rh.get_intersection_time(region, model_type)[1]]
        stock = model.ftrans(model.nom(year_range))
        sales = np.diff(stock)
        df = pd.DataFrame({"stock": stock[:-1], "sales": sales}, index=year_range[:-1])
        return df["sales"].max()
    except:
        return np.nan


# +
res_dict = {}

for region in peak_values_by_region.index:
    for model_type in rh.get_model_types():
        res_dict[(region, model_type)] = get_sales_max(region, model_type)

max_sales_ser = pd.Series(res_dict).rename("max_sales")
max_sales_ser.index = max_sales_ser.index.rename(["region", "model_type"])

# +
df = pd.concat(
    [
        pd.concat(
            [
                max_sales_ser,
                max_sales_ser.div(peak_values_by_region, level="region").rename(
                    "factor"
                ),
            ],
            axis=1,
        ).unstack(),
        peak_values_by_region.rename(("market_peak", "data")),
    ],
    axis=1,
)

df.rename_axis(["quantity", "model"], axis=1, inplace=True)

df.sort_index(axis=1, inplace=True)
# -

# #### Maximum necessary model sales volume normalized by historic peak sales volume

df[("factor", "bass")].sort_values()




