import re
import os
import copy
import inspect
import numpy as np
import pandas as pd
import geopandas as gpd
import pyomo.environ as pyo
from pyomo.environ import *
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm
from pyomo.opt import TerminationCondition
from mpisppy.opt.lshaped import LShapedMethod

#------------SET OF FUNCTIONS TO RETRIEVE OBJECTIVE, VARIABLES, PARAMETERS AND CONSTRAINTS NAME FROM AN ABSTRACT MODEL--------------
## Class to create an adjacency matrix from a list of coordinates
class AdjacencyMatrix:

    # ------------------------------------ define the constructor ------------------------------------
    def __init__(
        self,
        adj_matrix=None,
        coordinates: gpd.geoseries.GeoSeries = [],
        kind="geometric",
        epsg=32610,
        mode="distance",
    ):
        self.coordinates = coordinates
        self.mode = mode
        self.kind = kind
        self.epsg = epsg
        if adj_matrix is None:
            self.adjacency_matrix = self.create_adjacency_matrix()
        else:
            self.adjacency_matrix = adj_matrix

    # ------------------------- define the method to create the adjacency matrix ---------------------
    # create the adjacency matrix using the geometric distance
    def __geometric_distance(self, n):
        distances = np.zeros((n, n))
        rows, cols = np.indices(distances.shape)

        for i in range(n):
            diag_row_sup = np.diag(rows, k=i)
            diag_col_sup = np.diag(cols, k=i)

            diag_row_inf = np.diag(rows, k=-i)
            diag_col_inf = np.diag(cols, k=-i)

            d = (
                self.coordinates.distance(
                    self.coordinates.shift(-i).to_crs(epsg=self.epsg)
                )
                .dropna()
                .values
            )

            distances[diag_row_sup, diag_col_sup] = d
            distances[diag_row_inf, diag_col_inf] = d
        return distances

    # create the adjacency matrix using the OSRM API
    def create_adjacency_matrix(self):

        n = len(self.coordinates)

        if self.kind == "geometric":
            self.coordinates = self.coordinates.to_crs(epsg=self.epsg)
            distances = self.__geometric_distance(n)

        elif self.kind == "osrm":
            distances = np.zeros((n, n))

            # split the coordinates in chunks of 100
            if n > 100:
                chunks = n // 100
            else:
                chunks = 1

            if n % 100 != 0:
                chunks += 1

            if type(self.coordinates) == gpd.geoseries.GeoSeries:
                coordinates_list = [
                    [point.xy[0][0], point.xy[1][0]] for point in self.coordinates
                ]

            fix_url = "http://router.project-osrm.org/table/v1/driving/"

            for i in range(chunks):
                if i < chunks - 1:
                    source_indexes = list(range(0 + 100 * i, 100 + 100 * i))
                    shift = 100
                else:
                    source_indexes = list(range(0 + 100 * i, n))
                    shift = n % 100

                for j in range(chunks):
                    if j < chunks - 1:
                        destination_indexes = list(range(0 + 100 * j, 100 + 100 * j))
                    else:
                        destination_indexes = list(range(0 + 100 * j, n))

                    # define the destinations and sources URL
                    sources_url = "sources=" + ";".join(
                        str(k - 100 * i) for k in source_indexes
                    )
                    destinations_url = "destinations=" + ";".join(
                        str(k - 100 * j + shift) for k in destination_indexes
                    )

                    # define the coordinates URL
                    coordinates_url = "".join(
                        [
                            "".join(
                                str(
                                    str(
                                        [coordinates_list[i][0], coordinates_list[i][1]]
                                    )
                                )[1:-1].split()
                            )
                            + ";"
                            for i in source_indexes + destination_indexes
                        ]
                    )[:-1]

                    # call the OSMR API
                    r = requests.get(
                        fix_url
                        + coordinates_url
                        + f"?annotations={self.mode}&"
                        + sources_url
                        + "&"
                        + destinations_url
                    )
                    routes = json.loads(r.content)

                    if "message" in routes.keys():
                        print(
                            fix_url
                            + coordinates_url
                            + f"?annotations={self.mode}&"
                            + sources_url
                            + "&"
                            + destinations_url
                        )

                    distances[
                        0 + 100 * i : 100 + 100 * i, 0 + 100 * j : 100 + 100 * j
                    ] = np.array(routes[self.mode + "s"])

        return distances

    # ---------------------------------------- implement the methods to save and load the solution -----------------------------------
    # save the solution
    def save(self, file_name):
        with open(file_name, "wb") as f:
            dill.dump(self, f)

    # load the solution
    @staticmethod
    def load(file_name):
        with open(file_name, "rb") as f:
            fl = dill.load(f)

        return fl

def getNameSets(model):
    sets = []
    for member in inspect.getmembers(model):
        if re.search(r"<class 'pyomo.*Set",  str(type(member[1]))):
            sets.append(member[0])
    return sets


def getNameVariables(model):
    variables = []
    for member in inspect.getmembers(model):
        if re.search(r"<class 'pyomo.*Var",  str(type(member[1]))):
            variables.append(member[0])
    return variables


def getNameParameters(model):
    param = []
    for member in inspect.getmembers(model):
        if re.search(r"<class 'pyomo.*Param",  str(type(member[1]))):
            param.append(member[0])
    return param


def getNameConstraints(model):
    constraints = []
    for member in inspect.getmembers(model):
        if re.search(r"<class 'pyomo.*Constraint",  str(type(member[1]))):
            constraints.append(member[0])
    return constraints


def getNameExpressions(model):
    expressions = []
    for member in inspect.getmembers(model):
        if re.search(r"<class 'pyomo.*Expression",  str(type(member[1]))):
            expressions.append(member[0])
    return expressions


def getNameObjective(model):
    objective = []
    for member in inspect.getmembers(model):
        if re.search(r"<class 'pyomo.*Objective",  str(type(member[1]))):
            objective.append(member[0])
    return objective


#------------------------------------FUNCTION TO HELP IN GENEARTE SCENARIO ISTANCE-------------------------------------

def scenario_creator(scenarioName, model, scenarioData, scenarios, scenariosProbabilities, scenarioProbability, 
                        firstStageObjective, firstStageVariables, fixFirstStageVariables = {}):
    istance = model.create_instance(scenarioData(scenarioName, scenarios))
    
    if len(fixFirstStageVariables) != 0:
        for key in fixFirstStageVariables.keys():
            for key2 in fixFirstStageVariables[key]:
                istance.component(key)[key2].fix(fixFirstStageVariables[key][key2])

    #we specify which part of the objective function and which set of variables
    #belong to the first stage of the problem
    sputils.attach_root_node(istance, istance.component(firstStageObjective), [istance.component(v) for v in firstStageVariables])
    istance._mpisppy_probability = scenarioProbability(scenarioName, scenariosProbabilities)
    return istance


#------------------------------------FUNCTIONS TO COMPUTE MAIN EVALUATION METRICS-------------------------------------

#--------------------------------------- ----RP (recoursive solution)-------------------------------------------------

def RP_solution(options, all_scenario_names, scenario_creator_kwargs, method = 'EF', verbose = False):
    if method == 'EF':
        return RP_EF(options, all_scenario_names, scenario_creator_kwargs, verbose)

    elif method == 'LS':
        return RP_LS(options, all_scenario_names, scenario_creator_kwargs, verbose)

    else:
        print('only EF and LS available')


#-----------------Extensive Form---------------

def RP_EF(options, all_scenario_names, scenario_creator_kwargs, verbose):
    ef = ExtensiveForm(options, 
                all_scenario_names, 
                scenario_creator,
                scenario_creator_kwargs=scenario_creator_kwargs)

    results = ef.solve_extensive_form()

    terminationCondition =  results.Solver[0]['Termination condition']
    
    if verbose:
        objval = ef.get_objective_value()
        print(objval)

        soln = ef.get_root_solution()
        for (var_name, var_val) in soln.items():
            print(var_name, var_val)

    return ef, terminationCondition


#--------------------LShaped-------------------

def RP_LS(options, all_scenario_names, scenario_creator_kwargs, verbose = False):
    ls = LShapedMethod(options, 
                        all_scenario_names, 
                        scenario_creator,
                        scenario_creator_kwargs=scenario_creator_kwargs)

    result = ls.lshaped_algorithm()

    terminationCondition =  result.Solver[0]['Termination condition']

    if verbose:
        variables = ls.gather_var_values_to_rank0()
        for ((scen_name, var_name), var_value) in variables.items():
            print(scen_name, var_name, var_value)

    return ls, terminationCondition
    

#--------------------------------------------------WS (wait and see solution)-------------------------------------------

def WS_solution(options, all_scenario_names, scenario_creator_kwargs, method='EF', verbose = False):
    if method == 'EF':
        return WS_EF(options, all_scenario_names, scenario_creator_kwargs, verbose)

    elif method == 'LS':
        return WS_LS(options, all_scenario_names, scenario_creator_kwargs, verbose)

    else:
        print('only EF and LS available')


#--------------------LShaped-------------------

def WS_LS(options, all_scenario_names, scenario_creator_kwargs, verbose = False):
    det_solutions = {}
    scenario_creator_kwargs_copy = copy.deepcopy(scenario_creator_kwargs)
    scenario_creator_kwargs_copy['scenariosProbabilities'] = {name: 1 for name in all_scenario_names}

    for name in all_scenario_names:
        det_solutions[name] = RP_solution(options, [name], scenario_creator_kwargs_copy, 'LS', verbose)
    
    det_solutions['WS'] = np.mean([v[0].root.obj() for v in det_solutions.values()])
    return det_solutions

#-----------------Extensive Form---------------

def WS_EF(options, all_scenario_names, scenario_creator_kwargs, verbose = False):
    det_solutions = {}
    scenario_creator_kwargs_copy = copy.deepcopy(scenario_creator_kwargs)
    scenario_creator_kwargs_copy['scenariosProbabilities'] = {name: 1 for name in all_scenario_names}

    for name in all_scenario_names:
        det_solutions[name] = RP_solution(options, [name], scenario_creator_kwargs_copy, 'EF', verbose)
    
    det_solutions['WS'] = np.mean([v[0].get_objective_value() for v in det_solutions.values()])
    return det_solutions

#-------------------EVPI (expected value of perfect information)------------------

def EVPI_value(options, all_scenario_names, scenario_creator_kwargs, method = 'EF', verbose = False):
    if method == 'EF':
        RP_EF = RP_solution(options, all_scenario_names, scenario_creator_kwargs, 'EF', verbose)
        RP = RP_EF[0].get_objective_value()
        if RP_EF[1] == TerminationCondition.infeasible:
            return 'RP solution is infeasible'
    
        WS_dict = WS_solution(options, all_scenario_names, scenario_creator_kwargs, 'EF', verbose)
        WS = WS_dict['WS']
    
        for v in list(WS_dict.values())[:len(WS_dict)-1]:
            if v[1] == TerminationCondition.infeasible:
                return 'One of WS solution is infeasible'
    
    elif method == 'LS':
        RP_LS = RP_solution(options, all_scenario_names, scenario_creator_kwargs, 'LS', verbose)
        RP = RP_LS[0].root.obj()
        if RP_LS[1] == TerminationCondition.infeasible:
            return 'RP solution is infeasible'

        WS_dict = WS_solution(options, all_scenario_names, scenario_creator_kwargs, 'LS', verbose)
        WS = WS_dict['WS']
    
        for v in list(WS_dict.values())[:len(WS_dict)-1]:
            if v[1] == TerminationCondition.infeasible:
                return 'One of WS solution is infeasible'

    return abs(WS-RP)


#-------------------EV (expected value solution)------------------
def EV_solution(options, scenario_creator_kwargs, method = 'EF', verbose = False):
    scenario_creator_kwargs1 = copy.deepcopy(scenario_creator_kwargs)
    scenarios = scenario_creator_kwargs['scenarios']

    scenariosProbabilities = scenario_creator_kwargs['scenariosProbabilities']
    scenarioProbability = scenario_creator_kwargs['scenarioProbability']

    if type(scenarios) == dict:
        adj_matricies = []
        for v in scenarios.values():
            adj_matricies.append(v.adjacency_matrix)
        
        avg_matrix = np.mean(adj_matricies, axis=0)
        scenario_creator_kwargs1['scenarios']["avg_scenario"] = AdjacencyMatrix(avg_matrix)

    else: 
        print('scenarios must be a dictionary')

    scenario_creator_kwargs1['scenariosProbabilities']["avg_scenario"] = 1

    if method == 'LS':
        options_copy = copy.deepcopy(options)
        if 'valid_eta_lb' in options.keys():
            options_copy.pop('valid_eta_lb')


    if method == 'EF':
        EV = RP_solution(options, ["avg_scenario"], scenario_creator_kwargs1, 'EF', verbose)
    elif method == 'LS':
        EV = RP_solution(options_copy, ["avg_scenario"], scenario_creator_kwargs1, 'LS', verbose)

    return EV




#-------------------EEV (Expectation of the expected solution)------------------

def EEV_solution(options, all_scenario_names, scenario_creator_kwargs, method = 'EF', verbose = False):
    
    if method == 'EF':
        EV = EV_solution(options, scenario_creator_kwargs, 'EF', verbose)
        variables = EV[0].ef.component_objects(pyo.Var)


    elif method == 'LS':
        EV = EV_solution(options, scenario_creator_kwargs, 'LS', verbose)
        variables = EV[0].root.component_objects(pyo.Var)

    if EV[1] == TerminationCondition.infeasible:
        return 'EV is infeasible'

    fixFirstStageVariables = {}
    for var in scenario_creator_kwargs['firstStageVariables']:
        for v in variables:
            if var in str(v):       
                fixFirstStageVariables[var] = {}
                for index in v:
                    fixFirstStageVariables[var][index] = pyo.value(v[index])
    
    scenario_creator_kwargs1 = copy.deepcopy(scenario_creator_kwargs)
    scenario_creator_kwargs1['fixFirstStageVariables'] = fixFirstStageVariables

    if 'valid_eta_lb' in scenario_creator_kwargs.keys():
        scenario_creator_kwargs1.pop('valid_eta_lb')

    return RP_solution(options, all_scenario_names, scenario_creator_kwargs1, method, verbose)

    
#-------------------VSS (value of stochastic solution)------------------

def VSS_value(options, all_scenario_names, scenario_creator_kwargs, method = 'EF', verbose = False, RP = None):
    EVV = EEV_solution(options, all_scenario_names, scenario_creator_kwargs, method, verbose)
    if type(EVV) == str:
        return 'EV is infeasible'
    elif EVV[1] == TerminationCondition.infeasible:
        return 'EVV is infeasible'

    if RP is None:
        RP = RP_solution(options, all_scenario_names, scenario_creator_kwargs, method, verbose)
        if RP[1] == TerminationCondition.infeasible:
            return 'RP is infeasible'
        if method == 'EF':
            RP_value = RP[0].get_objective_value()
        elif method == 'LS':
            RP_value = RP[0].root.obj()
    else:
        RP_value = RP.solution_value
    
    if method == "EF":
        EVV_value = EVV[0].get_objective_value()
    elif method == "LS":
        EVV_value = EVV[0].root.obj()
    
    VSS = abs(RP_value-EVV_value)

    return VSS


#-------------------Out-sample evaluation------------------

def Out_of_sample_evaluation(options, all_scenario_names, scenario_creator_kwargs, variables, method = 'EF', verbose = False):
    
    if type(variables) == dict:
        fixFirstStageVariables = variables
    else:
        fixFirstStageVariables = {}
        for var in scenario_creator_kwargs['firstStageVariables']:
            for v in variables:
                if var in str(v):       
                    fixFirstStageVariables[var] = {}
                    for index in v:
                        fixFirstStageVariables[var][index] = pyo.value(v[index])
    
    scenario_creator_kwargs1 = copy.deepcopy(scenario_creator_kwargs)
    scenario_creator_kwargs1['fixFirstStageVariables'] = fixFirstStageVariables

    if 'valid_eta_lb' in scenario_creator_kwargs.keys():
        scenario_creator_kwargs1.pop('valid_eta_lb')

    return RP_solution(options, all_scenario_names, scenario_creator_kwargs1, method, verbose)



#------------------------------------FUNCTIONS TO COMPUTE MAIN EVALUATION METRICS-------------------------------------

def evaluate_stochastic_solution(options, scenario_creator_kwargs, all_scenario_names, RP=None, fls_deterministics=None, method = 'EF',
                                df = pd.DataFrame(columns=['n_locations', 'RP', 'RP_Out_of_Sample', 'WS', 'EVPI', 'VSS']), lb = 0):
    options_copy = copy.deepcopy(options)

    if method == 'LS' and 'valid_eta_lb' in options.keys():
        bounds = {name: lb for name in all_scenario_names}
        options_copy['valid_eta_lb'] = bounds
    
    ## ------------------- Load or Solve the RP optimization model-------------------
    if RP is not None:
        RP_value = RP.solution_value
        n_locations = RP.n_of_locations_to_choose
        fixedVar = {"x": {idx: round(RP.first_stage_solution[idx],0) for idx in RP.first_stage_solution}}
    else:
        RP = RP_solution(options_copy, all_scenario_names, scenario_creator_kwargs, method)
        if method == 'EF':
            RP_value = RP[0].get_objective_value()
            fixedVar = RP[0].ef.component_objects(pyo.Var)
            
        elif method == 'LS':
            RP_value = RP[0].root.obj()
            fixedVar = RP[0].root.component_objects(pyo.Var)
    
    ## ------------------- Compute the WS value -------------------
    if fls_deterministics is not None:
        WS = sum([fls_deterministics[scenario].solution_value*scenario_creator_kwargs["scenariosProbabilities"][scenario] for scenario in all_scenario_names])
    else:
        WS = WS_solution(options_copy, all_scenario_names, scenario_creator_kwargs, method)['WS']
    
    ## ------------------- Compute the EVPI value -------------------
    EVPI = abs(WS-RP_value)
    
    ## ------------------- Compute the VSS value -------------------
    VSS = VSS_value(options_copy, all_scenario_names, scenario_creator_kwargs, method, RP = RP)
    
    ## ------------------- Compute the Out of Sample value -------------------
    RP_Out_of_Sample_value = "Not computed"
    
    # TODO : extract scenarios needed for the out of sample evaluation
    # RP_Out_of_Sample = Out_of_sample_evaluation(options_copy, 
    #                                                 all_scenario_names,
    #                                                 scenario_creator_kwargs,
    #                                                 fixedVar,
    #                                                 method)

    # if type(RP_Out_of_Sample) == str:
    #     RP_Out_of_Sample_value = RP_Out_of_Sample
    # else:
    #     if method == 'EF':  
    #         RP_Out_of_Sample_value = RP_Out_of_Sample[0].get_objective_value()
    #     if method == 'LS':
    #         RP_Out_of_Sample_value = RP_Out_of_Sample[0].root.obj()


    df = pd.concat([df, pd.DataFrame({'n_locations': [n_locations], 'RP': [RP_value], 'RP_Out_of_Sample' : [RP_Out_of_Sample_value], 'WS': [WS], 'EVPI':[EVPI], 'VSS':[VSS]})], ignore_index=True)
    return df