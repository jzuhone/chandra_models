
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
    ''' Wrapper for running a Xija model and simplifying the setting of initial conditions.

    Example1::
      >>> import chandra_models
      >>> dpa = chandra_models.XijaModelRun('dpa', tstart='2014:001', tstop='2014:030')
      >>> plot(dpa.tlm['date'], dpa.model['dpa])

    Example2::
      >>> import chandra_models
      >>> init_dict = {"pitch":{"value":100}, "pf0tank2t":{"value":75}}
      >>> tank = chandra_models.XijaModelRun('pftank2t', tstart='2014:001', tstop='2014:030',
                                             initial_values=init_dict)
      >>> plot(tank.tlm['date'], tank.model['dpa])

    All Xija models defined in ``chandra_models`` are supported.



    TODO: ****************focal plane model not supported yet****************



    :param model_name: name of model
    :param tstart: start time of model run
    :param tstop: stop time of model run
    :param initial_values: dictionary of initial values, see above for an example

    :returns: XijaModelRun object containing the relevant model and telemetry data along with all
              inputs

    Notes:
    Temperatures are input according to their engineering units, as specified in the meta data
    file (read into self.model_info).


   '''


    def __init__(self, model_name, tstart="2012:001", tstop="2012:100", initial_values=None):

        self.model_name = model_name
        self.initial_values = initial_values
        self.tstop = DateTime(tstop).secs
        self.tstart = DateTime(tstart).secs

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
        ''' Convert calculated temps to their common engineering units (Celsius to Fahrenheit).

        This converts the model outputs, which are always in Celsius, to their engineering units.
        At this time, this only converts to Fahrenheit.
        '''
        for node in self.model_info["msids"].keys():
            if self.msids[node]["units"].lower() == "degf":
                self.model[node] = C2F(self.model[node])
                self.tlm[node] = C2F(self.tlm[node])

        for node in self.pseudo_msids.keys():
            if self.model_info["pseudo_msids"][node]["units"].lower() == "degf":
                self.model[node] = C2F(self.model[node])

    def __process_initial_values_helper(self, node, datastructure):
        ''' Helper function that stores initial values in their appropriate locations, if specified.

        This should work for msids, pseudo msids, and state values.
        '''
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
        ''' Assign initial values, if specifed in initial_values keyword.

        These values are stored in pre-defined datastructures.
        '''

        for key in self.initial_values.keys():

            if key in self.msids.keys():
                self.__process_initial_values_helper(key, self.msids)

            if key in self.pseudo_msids.keys():
                self.__process_initial_values_helper(key, self.pseudo_msids)

            if key in self.state_cols.keys():
                self.__process_initial_values_helper(key, self.state_cols)

    def _convert_initial_units(self):
        ''' Convert all initial values or offsets to Celsius.

        This includes all default initial values or offsets, or initial values set manually via
        the initial_values keyword.

        This works only on msids and pseudo-msids, no conversions for state_cols at this time.

        '''
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
        ''' Initialize the nodes in the xijamodel object, if initial values are specified.

        Since all pseudo-msids need to be initialized, for all pseudo-msids, if an initial value
        is specified, it is used, otherwise the initial value is calculated based on an offset
        from the associated node (e.g. tcylaft6_0 is set based on tcylaft6).
        '''
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

        for node in self.pseudo_msids.keys():
            override = self.pseudo_msids[node]["initialization"]["value"]
            if not isinstance(override, (int, long, float, list, np.ndarray)):
                offset = self.pseudo_msids[node]["initialization"]["offset"]
                assoc_node = self.pseudo_msids[node]["initialization"]["offset_msid"]
                assoc_node_init_value = xijamodel.get_comp(assoc_node).dvals[0]
                xijamodel.comp[node].set_data(assoc_node_init_value + offset)
            else:
                xijamodel.comp[node].set_data(override)

    def _get_model_output(self, xijamodel):
        ''' Return the calculated values along with associated telemetry. Any telemetry mnemonic 
        that is set via an initial value (or array of values) will only reflect the value(s)
        manually set (i.e. not values from the engineering archive).
        '''

        model = {}

        tlm = {"date": xijamodel.get_comp(self.model_info["msids"].keys()[0]).times}

        for node in self.model_info["msids"].keys():
            model[node] = xijamodel.get_comp(node).mvals
            tlm[node] = xijamodel.get_comp(node).dvals

        for node in self.model_info["pseudo_msids"].keys():
            model[node] = xijamodel.get_comp(node).mvals

        self.model = model
        self.tlm = tlm

    def _calcmodel(self):
        ''' Core xija routine for calculating model values.
        '''
        tstart = DateTime(self.tstart).date
        tstop = DateTime(self.tstop).date

        xijamodel = xija.ThermalModel(self.model_name, start=tstart, stop=tstop,
                                      model_spec=self.pars)
        self._init_nodes(xijamodel)
        xijamodel.make()
        xijamodel.calc()
        self._get_model_output(xijamodel)
