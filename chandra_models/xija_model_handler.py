
import json
import numpy as np

from Chandra.Time import DateTime
import xija
import chandra_models


def K2F(k):
    return (k - 273.15) * 1.8 + 32.


def C2F(k):
    return k * 9 / 5 + 32.


def F2C(k):
    return (k - 32.) * 5 / 9


class XijaModelRun(object):

    def __init__(self, model_name, initial_values=None, tstop="2012:001", dt=328, numdays=30):
        """
        Run Xija Model

        init = {"pitch":{"value":v, "times":t}}
        """
        self.tstop = DateTime(tstop).secs
        self.numdays = numdays
        self.dt = dt
        self.tstart = DateTime(tstop).secs - numdays * 24 * 3600
        self.model_name = model_name
        self.initial_values = initial_values
        self.pars, self.model_info = chandra_models.get_xija_model_spec(model_name, meta=True)

        self.state_cols = self.model_info["state_cols"]
        self.msids = self.model_info["msids"]
        self.pseudo_msids = self.model_info["pseudo_msids"]
        if self.initial_values:
            self._process_initial_values()

        self._convert_initial_units()
        self._calcmodel()
        self._convert_output_units()

    def _convert_output_units(self):

        for node in self.model_info["msids"].keys():
            if self.msids[node]["units"].lower() == "degf":
                self.model[node] = C2F(self.model[node])
                self.tlm[node] = C2F(self.tlm[node])

        for node in self.pseudo_msids.keys():
            if self.model_info["pseudo_msids"][node]["units"].lower() == "degf":
                self.model[node] = C2F(self.model[node])

    def __process_initial_values_helper(self, node, datastructure):
        if node in datastructure.keys():

            value = self.initial_values[node]["value"]
            if isinstance(value, list):
                value = np.array(value)
            datastructure[node]["initialization"]["value"] = value

            if "times" in self.initial_values[node]:
                if isinstance(self.initial_values[node]["times"], (int, long, float, list, np.ndarray)):
                    times = self.initial_values[node]["times"]
                    if isinstance(value, list):
                        times = np.array(times)
                    datastructure[node]["initialization"]["times"] = times

    def _process_initial_values(self):

        for key in self.initial_values.keys():

            if key in self.msids.keys():
                self.__process_initial_values_helper(key, self.msids)

            if key in self.model_info["pseudo_msids"].keys():
                self.__process_initial_values_helper(key, self.pseudo_msids)

            if key in self.model_info["state_cols"].keys():
                self.__process_initial_values_helper(key, self.state_cols)

    def _convert_initial_units(self):
        # No conversions for state_cols at this time
        for node in self.msids.keys():
            if self.msids[node]["units"].lower() == "degf":
                if isinstance(self.msids[node]["initialization"]["value"], (int, long, float,
                                                                            list, np.ndarray)):
                    init_value = self.msids[node]["initialization"]["value"]
                    self.msids[node]["initialization"]["value"] = F2C(init_value)

        for node in self.pseudo_msids.keys():
            if self.pseudo_msids[node]["units"].lower() == "degf":
                if isinstance(self.pseudo_msids[node]["initialization"]["value"], (int, long,
                                                                                   float, list, np.ndarray)):
                    init_value = self.pseudo_msids[node]["initialization"]["value"]
                    self.pseudo_msids[node]["initialization"]["value"] = F2C(init_value)
                else:
                    offset = self.pseudo_msids[node]["initialization"]["offset"]
                    self.pseudo_msids[node]["initialization"]["offset"] = offset * 5 / 9

    def _init_nodes(self, xijamodel):
        if self.initial_values:
            for node in self.initial_values.keys():

                if node in self.msids.keys():
                    init_data = self.msids[node]["initialization"]
                    if not isinstance(init_data["times"], (int, long, float, list, np.ndarray)):
                        init_data["times"] = None
                    xijamodel.comp[node].set_data(init_data["value"], times=init_data["times"])

                if node in self.state_cols.keys():
                    init_data = self.state_cols[node]["initialization"]
                    if not isinstance(init_data["times"], (int, long, float, list, np.ndarray)):
                        init_data["times"] = None
                    xijamodel.comp[node].set_data(init_data["value"], times=init_data["times"])

        # We don"t ever specify pseudo msid times, so disregard this capability for now
        for node in self.pseudo_msids.keys():
            override = self.pseudo_msids[node]["initialization"]["value"]
            if not isinstance(override, (int, long, float, list, np.ndarray)):
                offset = self.pseudo_msids[node]["initialization"]["offset"]
                assoc_node = self.pseudo_msids[node]["initialization"]["offset_msid"]
                assoc_node_init_value = xijamodel.get_comp(assoc_node).dvals[0]
                xijamodel.comp[node].set_data(assoc_node_init_value + offset)
            else:
                xijamodel.comp[node].set_data(override)

    def _calcmodel(self):
        tstart = DateTime(self.tstart).date
        tstop = DateTime(self.tstop).date

        xijamodel = xija.ThermalModel(self.model_name, start=tstart, stop=tstop,
                                      model_spec=self.pars)
        self._init_nodes(xijamodel)
        xijamodel.make()
        xijamodel.calc()
        self._get_model_output(xijamodel)

    def _get_model_output(self, xijamodel):

        model = {}

        tlm = {"date": xijamodel.get_comp(self.model_info["msids"].keys()[0]).times}

        for node in self.model_info["msids"].keys():
            model[node] = xijamodel.get_comp(node).mvals
            tlm[node] = xijamodel.get_comp(node).dvals

        for node in self.model_info["pseudo_msids"].keys():
            model[node] = xijamodel.get_comp(node).mvals

        self.model = model
        self.tlm = tlm
