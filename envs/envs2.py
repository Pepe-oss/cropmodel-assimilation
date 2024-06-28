from utils import read_from_input_file

import gymnasium as gym
import pandas as pd
import numpy as np
import random
import os

import pcse

class ModelRerunner(object):
    """Reruns a given model with different values of parameters TWDI and SPAN.

    Returns a pandas DataFrame with simulation results of the model with given
    parameter values.
    """

    def __init__(self, params, wdp, agro):
        self.params = params
        self.wdp = wdp
        self.agro = agro

    def __call__(self, par_values):
        # Check if correct number of parameter values were provided
        #         if len(par_values) != len(self.parameters):
        #             msg = "Optimizing %i parameters, but only % values were provided!" % (len(self.parameters, len(par_values)))
        #             raise RuntimeError(msg)
        #         # Clear any existing overrides
        #         self.params.clear_override()
        # Set overrides for the new parameter values
        self.params['SPAN'] = par_values[0]
        self.params['KDIFTB'][1] = par_values[1]
        self.params['KDIFTB'][3] = par_values[2]
        self.params['EFFTB'][1] = par_values[3]
        self.params['EFFTB'][3] = par_values[4]
        self.params['AMAXTB'][1] = par_values[5]
        self.params['AMAXTB'][3] = par_values[6]
        self.params['AMAXTB'][5] = par_values[7]
        self.params['CVS'] = par_values[8]
        self.params['CVO'] = par_values[9]
        self.params['CVL'] = par_values[10]
        self.params['CVR'] = par_values[11]

        # Run the model with given parameter values
        self.wofost = pcse.models.Wofost72_WLP_FD(self.params, self.wdp, self.agro)
        self.wofost.run_till_terminate()
        df = pd.DataFrame(self.wofost.get_output()).set_index("day")
        return df

class ObjectiveFunctionCalculator(object):
    """Computes the objective function.

    This class runs the simulation model with given parameter values and returns the objective
    function as the sum of squared difference between observed and simulated LAI.
.   """

    def __init__(self, params, wdp, agro, observations):
        self.modelrerunner = ModelRerunner(params, wdp, agro)
        self.df_observations = observations
        self.n_calls = 0

    def __call__(self, par_values, grad=None):
        """Runs the model and computes the objective function for given par_values.

        The input parameter 'grad' must be defined in the function call, but is only
        required for optimization methods where analytical gradients can be computed.
        """
        self.n_calls += 1
        # print(".", end="")
        # Run the model and collect output
        self.df_simulations = self.modelrerunner(par_values)
        # compute the differences by subtracting the DataFrames
        # Note that the dataframes automatically join on the index (dates) and column names
        df_differences = self.df_simulations - self.df_observations
        # Compute the RMSE on the LAI column
        obj_func = -np.sqrt(np.mean(df_differences.LAI**2))
        return df_differences, obj_func

class PcseRLAssimilationEnv(gym.Env):

    _PATH_TO_FILE = os.path.realpath('../../../input')

    _DEFAULT_CONFIG = 'Wofost72_WLP_FD.conf'

    def __init__(self,
                 model_config: str = _DEFAULT_CONFIG,
                 file_config: str = _PATH_TO_FILE,
                 seed: int = None,
                 **kwargs
                 ):

        # Optionally set the seed
        super().reset(seed=seed)

        (cropd, soild, self._agro_management,
         self._weather_data_provider) = read_from_input_file(file_config)
        site = pcse.util.WOFOST72SiteDataProvider(WAV=30)

        # Store the crop/soil/site parameters
        self._crop_params = cropd
        self._site_params = site
        self._soil_params = soild
        # Combine the config files in a single PCSE ParameterProvider object
        self._parameter_provider = pcse.base.ParameterProvider(cropdata=self._crop_params,
                                                               sitedata=self._site_params,
                                                               soildata=self._soil_params,
                                                               )

        # Store the PCSE Engine config
        self.df_pseudo_obs = self._get_obs()

        # Create a PCSE engine / crop growth model
        self._model = self._init_pcse_model()

        # Use the config files to extract relevant settings
        self._output_variables = ["DVS","LAI","TAGP", "TWSO", "TWLV", "TWST",
                                  "TWRT", "TRA", "RD", "SM", "WWLOW"]
        self._weather_variables = list(pcse.base.weather.WeatherDataContainer.required)

        # Define Gym observation space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,))
        # Define Gym action space
        self.action_space = self._get_action_space()

    def _init_pcse_model(self, *args, **kwargs):

        # Create a PCSE engine / crop growth model
        model = ObjectiveFunctionCalculator(self._parameter_provider,
                                            self._weather_data_provider,
                                            self._agro_management,
                                            self.df_pseudo_obs)

        self._init_firstguess = self._parameter_provider_init()

        return model

    def _parameter_provider_init(self):
        return [self._parameter_provider['SPAN'],
                self._parameter_provider['KDIFTB'][1],
                self._parameter_provider['KDIFTB'][3],
                self._parameter_provider['EFFTB'][1],
                self._parameter_provider['EFFTB'][3],
                self._parameter_provider['AMAXTB'][1],
                self._parameter_provider['AMAXTB'][3],
                self._parameter_provider['AMAXTB'][5],
                self._parameter_provider['CVS'],
                self._parameter_provider['CVO'],
                self._parameter_provider['CVL'],
                self._parameter_provider['CVR']
                ]

    def _get_action_space(self) -> gym.spaces.Space:
        lower_bound = [28,0.4,0.4,0.4,0.4,38,38,38,0.66,0.66,0.66,0.66]
        upper_bound = [33,0.7,0.7,0.5,0.7,45,45,45,0.9,0.9,0.9,0.9]

        space = gym.spaces.Box(low=np.array(lower_bound), high=np.array(upper_bound), shape=(len(lower_bound),))

        return space

    def _get_obs(self):
        # df_pseudo_obs = pd.read_excel(file)
        # df_pseudo_obs.index = pd.to_datetime(df_pseudo_obs.DateMeas)

        model_runner = ModelRerunner(self._parameter_provider,
                                     self._weather_data_provider,
                                     self._agro_management)
        df = model_runner(self._random_action())
        df = self._find_closest_lai_values(df)

        return df

    def _find_closest_lai_values(self, df, dvs_column='DVS', lai_column='LAI'):
        # 移除 DVS 列中的 NA 值
        non_na_df = df.dropna(subset=[dvs_column])

        target_values = [0.35, 0.62, 0.82, 0.92, 1.04, 1.16, 1.25, 1.33, 1.45, 1.60, 1.74]

        # 初始化一个空列表来保存结果
        closest_indices_and_lai = []

        # 遍历目标值列表
        for target in target_values:
            # 找到 DVS 列中与 target 最接近的值（不包括 NA）
            closest_index = (non_na_df[dvs_column] - target).abs().idxmin()
            # 获取 LAI 列的相应值
            closest_lai = df.loc[closest_index, lai_column]
            # 将索引和 LAI 值作为元组添加到结果列表中
            closest_indices_and_lai.append((closest_index, closest_lai))

            # 创建一个新的 DataFrame，索引为原始索引，数据为 LAI 值
        # 使用字典推导式将元组列表转换为字典
        closest_df = pd.DataFrame({lai_column: [item[1] for item in closest_indices_and_lai]},
                                  index=[item[0] for item in closest_indices_and_lai])
        closest_df.index.name = 'day'

        return closest_df

    def _random_action(self):
        lower_bound = [28,0.4,0.4,0.4,0.4,38,38,38,0.66,0.66,0.66,0.66]
        upper_bound = [33,0.7,0.7,0.5,0.7,45,45,45,0.9,0.9,0.9,0.9]

        ls = []
        for i in range(len(lower_bound)):
            ls.append(random.uniform(lower_bound[i], upper_bound[i]))
        return ls

    def step(self, action) -> tuple:
        # Create a dict for storing info
        info = dict()

        # Construct an observation and reward from the new environment state
        df_differences, r = self._apply_action(action)

        o = self._get_observation(df_differences)

        done = False
        truncated = False
        guess = self._parameter_provider_init()
        # print(r)
        if r>=-0.5: done = True

        info["observation"] = o
        info["reward"] = r

        return o, r, done, truncated, info

    def _apply_action(self, action):
        return self._model(action)

    def _get_observation(self, df_differences):
        return np.array(df_differences.LAI.dropna().values)

    def reset(self,
              *,
              seed: int = None,
              return_info: bool = False,
              options: dict = None
              ):
        """
        Reset the PCSE-Gym environment to its initial state

        :param seed:
        :param return_info: flag indicating whether an info dict should be returned
        :param options: optional dict containing options for reinitialization
        :return: depending on the `return_info` flag, an initial observation is returned or a two-tuple of the initial
                 observation and the info dict
        """
        # print('new')
        # Optionally set the seed
        super().reset(seed=seed)

        # Create an info dict
        info = dict()

        # Create a PCSE engine / crop growth model
        self.df_pseudo_obs = self._get_obs()
        df_differences, _ = self._apply_action(self._init_firstguess)
        o = self._get_observation(df_differences)

        guess = self._parameter_provider_init()
        if self._init_firstguess != guess: return_info = True

        info['observation'] = o

        return o, info if return_info else o

    def render(self, mode="human"):
        pass  # Nothing to see here