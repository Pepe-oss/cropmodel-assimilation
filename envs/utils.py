import os
import yaml
from dotmap import DotMap
import matplotlib.pyplot as plt

from pcse.fileinput import CABOFileReader
from pcse.fileinput import ExcelWeatherDataProvider

def read_from_input_file(input_file):
    for root, dirs, files in os.walk(input_file):
        for file in files:
            if file.endswith(".crop"):
                crop_parameters = os.path.join(root, file)
            elif file.endswith(".soil"):
                soil_parameters = os.path.join(root, file)
            elif file.endswith(".agro"):
                agro_parameters = os.path.join(root, file)
            elif file.endswith(".xlsx"):
                met_parameters = os.path.join(root, file)

    # If any parameter files are specified as path, convert them to a suitable object for pcse
    if isinstance(crop_parameters, str):
        crop_parameters = CABOFileReader(crop_parameters)
    if isinstance(soil_parameters, str):
        soil_parameters = CABOFileReader(soil_parameters)
    if isinstance(met_parameters, str):
        weather_data_provider = ExcelWeatherDataProvider(met_parameters)
    if isinstance(agro_parameters,str):
        with open(agro_parameters, 'r') as f:
            agro_management = yaml.load(f, Loader=yaml.SafeLoader)

    return crop_parameters, soil_parameters, agro_management, weather_data_provider

def make_plots(df, df_obs, df_compare=None):
    """A function for plotting data

    :param df: a dataframe with WOFOST simulation results
    :param df_obs: a dataframe with potato experimental data
    :param df: a dataframe with WOFOST simulation results to compare with
    """
    plots = [DotMap(title="Total crop biomass", variable="TAGP", observation="TotalDM", units="kg/ha"),
             DotMap(title="Leaf biomass", variable="TWLV", observation="LeafDM", units="kg/ha"),
             DotMap(title="Stem biomass", variable="TWST", observation="StemDM", units="kg/ha"),
             DotMap(title="Tuber biomass", variable="TWSO", observation="TuberDM", units="kg/ha"),
             DotMap(title="Leaf area index", variable="LAI", observation="LAI", units="-")
             ]
    fig, axes = plt.subplots(figsize=(16,14), nrows=3, ncols=2, sharex=True)
    for plot, ax in zip(plots, axes.flatten()):
        df[plot.variable].plot(ax=ax)
        df_obs[plot.observation].plot(ax=ax, marker="o", linestyle="")
        if df_compare is not None:
            df_compare[plot.variable].plot(ax=ax, linestyle=":")
        ax.set_title("%s [%s]" % (plot.title, plot.units))
        ax.set_ylabel(plot.units)
    fig.autofmt_xdate()
    fig.savefig("test.png", dpi=300)


