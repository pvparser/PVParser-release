import copy
import json
import multiprocessing
import time
from itertools import chain, combinations, islice, product
from math import comb

import numpy as np
from scipy.optimize import curve_fit

import ipal_iids.settings as settings
from ids.GeCo.equations import equations, get_equation
from ids.ids import MetaIDS

_DATAPOINTS = {}  # Shared global variable for faster multiprocessing
_EPS = 0.0000000001


# Has to be a standalone method since otherwise multiprocessing/fork would try to
# clone the entire GeCo class for each subprocess
def _system_identification(args):
    global _DATAPOINTS
    target, equation, combination, include_target_self = args

    # Omit combinations that contain the target. In the original GeCo setting the
    # target's previous value is added explicitly below; in cross-only diagnostics
    # we keep the target out entirely.
    if target in combination:
        return None, None

    # 0) Prepare datapoints
    combination = (target,) + combination if include_target_self else tuple(combination)
    if len(combination) == 0:
        return None, None
    # Product needs at least two variables and a single-variable product duplicates Sum.
    if equation.name == "Product" and len(combination) < 2:
        return None, None

    x_data = np.array([_DATAPOINTS[s][:-1] for s in combination])
    y_data = np.array(_DATAPOINTS[target][1:])

    # 1) Fit parameters on first part of the data with curve_fit
    try:
        # 80% of data for fitting the parameters
        SPLIT = int(0.8 * len(y_data))
        x_train = np.array([x[:SPLIT] for x in x_data])
        y_train = np.array(y_data[:SPLIT])

        potp, _ = curve_fit(
            equation.calc,
            x_train,
            y_train,
            p0=equation.default_parameters(combination),
        )
    except RuntimeError as err:
        settings.logger.error(f"Error in curve_fit: {err}")
        return None, None

    # 2) Calculate errors on 100% of the data
    errors = equation.calc(x_data, *potp) - y_data

    # 3) calculate CUSUM threshold (max error)
    drift = np.mean(abs(errors)) + np.std(errors)
    cusum = 0
    threshold = 0
    for err in errors:
        cusum = max(cusum + abs(err) - drift, 0)
        threshold = max(threshold, cusum)

    result = {
        "error": np.mean(errors**2),  # MSE
        "equation": equation,
        "combination": combination,
        "parameters": potp,
        "drift": drift,
        "threshold": threshold,
    }

    return target, result


class GeCo(MetaIDS):
    _name = "GeCo"
    _description = "GeCo returns whether a sensor's/actuator's current value exceeds a threshold of deviation from predicted data over time observed in the training data and raises an alarm if any observation falls outside that range. The predictions of the data are based on expected correlations with other sensor/actuator data."
    _requires = ["train.state", "live.state"]
    _GeCo_default_settings = {
        "ignore": [],
        "max_formel_length": 3,
        "threshold_factor": 1.1,
        "cusum_factor": 2,
        "cpus": 1,
        "include_target_self": True,
    }

    _supports_preprocessor = False

    def __init__(self, name=None):
        super().__init__(name=name)

        self._add_default_settings(self._GeCo_default_settings)

        self.CI = {}
        self.last_value = {}
        self.cusum = {}

    def _all_combinations(self, sensor_names):
        # All combinations of sensors up to max_formel_length
        f_len = range(0, self.settings["max_formel_length"] + 1)
        tests = chain(*[combinations(sensor_names, r) for r in f_len])

        # Calculate number of combinations
        n = len(sensor_names)
        count = sum([comb(n, r) for r in f_len])

        # All combinations of sensors, equations, and combinations
        include_target_self = bool(self.settings.get("include_target_self", True))
        formulas = (
            (target, equation, combination, include_target_self)
            for target, equation, combination in product(sensor_names, equations, tests)
        )
        count = int(len(sensor_names) * len(equations) * count)

        return count, formulas

    def _batch(self, it, n):
        assert n > 1
        while batch := tuple(islice(iter(it), n)):
            yield batch

    def train(self, ipal=None, state=None):
        global _DATAPOINTS

        # Load training data
        settings.logger.info("Loading training data")
        with self._open_file(state) as f:
            states = [json.loads(line)["state"] for line in f.readlines()]

        sensor_names = [s for s in states[0] if s not in self.settings["ignore"]]

        # Set global variables
        _DATAPOINTS = {}
        for sensor in sensor_names:
            _DATAPOINTS[sensor] = [state[sensor] for state in states]
        del states  # Clear memory

        # Test data
        if len(sensor_names) - 1 < self.settings["max_formel_length"]:
            settings.logger.warning(
                f'Number of sensors ({len(sensor_names)}) is smaller than max_formel_length ({self.settings["max_formel_length"]}).'
            )
            settings.logger.warning(f"Setting max_formel_length to {len(sensor_names)}")
            self.settings["max_formel_length"] = len(sensor_names) - 1

        # Prepare CI mining
        count, formulas = self._all_combinations(sensor_names)
        settings.logger.info(
            f"Mining {len(sensor_names)} variables up to formel length {self.settings['max_formel_length']} resulting in {count} combinations."
        )

        # Start parallel mining
        settings.logger.info(f"Working with {self.settings['cpus']} threads")
        with multiprocessing.get_context("fork").Pool(self.settings["cpus"]) as p:
            N = 0
            start = time.time()
            last_report = time.time()

            for batch in self._batch(formulas, self.settings["cpus"] * 100):
                results = p.map(_system_identification, batch)

                for target, res in results:
                    N += 1
                    if res is None:
                        continue

                    # check if found formula is new or better
                    if target not in self.CI or res["error"] < self.CI[target]["error"]:
                        self.CI[target] = res

                # Print progress at least every 60s
                now = time.time()
                if now - last_report >= 60:
                    # Save progress
                    self.save_trained_model(incomplete=True)

                    # Calculate ETA
                    last_report = now
                    ETA = (now - start) / (N / count) - (now - start)
                    settings.logger.info(
                        f"{N}/{count} ({100 * N / count:.3f}%) in {(now - start) / 3600:.2f}h (ETA {ETA / 3600:.2f}h)"
                    )

    def new_state_msg(self, msg):
        alert = False
        state = msg["state"]

        # Initialize data with the first data
        if len(self.cusum) == 0:
            self.cusum = {s: 0 for s in self.CI}
            self.last_value = {s: state[s] for s in state}
            return False, [self.cusum[s] for s in self.CI]

        # Perform prediction and CUSUM per sensor
        for sensor, ci in self.CI.items():
            # Predict diff
            X = [self.last_value[s] for s in ci["combination"]]
            estimate = ci["equation"].calc(X, *ci["parameters"])

            # Get real_diff from last values
            diff = estimate - state[sensor]

            # Calculate CUSUM and cap CUSUM to a maximum
            self.cusum[sensor] = max(
                self.cusum[sensor] + abs(diff) - ci["drift"] - _EPS,
                0,
            )
            self.cusum[sensor] = min(
                self.cusum[sensor],
                ci["threshold"] * self.settings["threshold_factor"]
                + ci["drift"] * self.settings["cusum_factor"]
                + _EPS,
            )

            if self.cusum[sensor] > ci["threshold"] * self.settings["threshold_factor"]:
                alert |= True

        # Update values
        self.last_value = state
        return alert, [self.cusum[s] for s in self.CI]

    def save_trained_model(self, incomplete=False):
        if self.settings["model-file"] is None:
            return False

        CI = copy.deepcopy(self.CI)
        for sensor in CI:
            CI[sensor]["equation"] = CI[sensor]["equation"].name
            CI[sensor]["parameters"] = list(CI[sensor]["parameters"])

        model = {
            "_name": self._name,
            "settings": self.settings,
            "incomplete": incomplete,
            "CI": CI,
        }

        with self._open_file(self._resolve_model_file_path(), mode="wt") as f:
            f.write(json.dumps(model, indent=4) + "\n")

        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        try:  # Open model file
            with self._open_file(self._resolve_model_file_path(), mode="rt") as f:
                model = json.load(f)
        except FileNotFoundError:
            settings.logger.info(
                "Model file {} not found.".format(str(self._resolve_model_file_path()))
            )
            return False

        # Test model
        assert self._name == model["_name"]
        assert (
            model["settings"]["max_formel_length"] == self.settings["max_formel_length"]
        )
        assert all(
            [
                a == b
                for a, b in zip(model["settings"]["ignore"], self.settings["ignore"])
            ]
        )

        if model["incomplete"]:
            settings.logger.warning("Model incomplete. Training did not finish?")
        if model["settings"]["cusum_factor"] != self.settings["cusum_factor"]:
            settings.logger.warning(
                "Using other cusum_factor than saved in previous model"
            )
        if model["settings"]["threshold_factor"] != self.settings["threshold_factor"]:
            settings.logger.warning(
                "Using other threshold_factor than saved in previous model"
            )

        # Load model
        self.CI = model["CI"]
        for sensor in self.CI:
            self.CI[sensor]["equation"] = get_equation(self.CI[sensor]["equation"])

        return True

    def visualize_model(self):
        for sensor, ci in self.CI.items():
            formula = ci["equation"].print(
                ci["combination"], ci["parameters"], precision=4
            )
            print(f"{sensor} = {formula}")

        return None, None
