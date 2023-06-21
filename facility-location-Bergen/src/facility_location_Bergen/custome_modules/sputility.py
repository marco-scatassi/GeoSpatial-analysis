import inspect
import re
from pyomo.environ import *
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm
import numpy as np
import copy
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import pandas as pd
from mpisppy.opt.lshaped import LShapedMethod

#------------SET OF FUNCTIONS TO RETRIEVE OBJECTIVE, VARIABLES, PARAMETERS AND CONSTRAINTS NAME FROM AN ABSTRACT MODEL--------------

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
    scenario_creator_kwargs_copy['scenariosProbabilities'] = [1]*len(all_scenario_names)

    for name in all_scenario_names:
        det_solutions[name] = RP_solution(options, [name], scenario_creator_kwargs_copy, 'LS', verbose)
    
    det_solutions['WS'] = np.mean([v[0].root.obj() for v in det_solutions.values()])
    return det_solutions

#-----------------Extensive Form---------------

def WS_EF(options, all_scenario_names, scenario_creator_kwargs, verbose = False):
    det_solutions = {}
    scenario_creator_kwargs_copy = copy.deepcopy(scenario_creator_kwargs)
    scenario_creator_kwargs_copy['scenariosProbabilities'] = [1]*len(all_scenario_names)

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
        for k in scenarios.keys():
            scenario_creator_kwargs1['scenarios'][k].append(
                sum([scenarios[k][i]*scenarioProbability(str(i), scenariosProbabilities) for i in range(len(scenarios[k]))]))

    else: 
        print('scenarios must be a dictionary')

    scenario_creator_kwargs1['scenariosProbabilities'].append(1)

    if method == 'LS':
        if 'valid_eta_lb' in options.keys():
            options_copy = copy.deepcopy(options)
            options_copy.pop('valid_eta_lb')


    if method == 'EF':
        EV = RP_solution(options, [str(len(scenarios))], scenario_creator_kwargs1, 'EF', verbose)
    elif method == 'LS':
        EV = RP_solution(options_copy, [str(len(scenarios))], scenario_creator_kwargs1, 'LS', verbose)

    return EV




#-------------------EEV (Expectation of the expected solution)------------------

def EVV_solution(options, all_scenario_names, scenario_creator_kwargs, method = 'EF', verbose = False):
    
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
    EVV = EVV_solution(options, all_scenario_names, scenario_creator_kwargs, method, verbose)
    if type(EVV) == str:
        return 'EV is infeasible'
    elif EVV[1] == TerminationCondition.infeasible:
        return 'EVV is infeasible'

    if RP is None:
        RP = RP_solution(options, all_scenario_names, scenario_creator_kwargs, method, verbose)
        if RP[1] == TerminationCondition.infeasible:
            return 'RP is infeasible'
    else:
        RP = RP
    
    if method == 'EF':
        VSS = abs(RP[0].get_objective_value()-EVV[0].get_objective_value())
    elif method == 'LS':
        VSS = abs(RP[0].root.obj()-EVV[0].root.obj())

    return VSS


#-------------------Out-sample evaluation------------------

def Out_of_sample_evaluation(options, all_scenario_names, scenario_creator_kwargs, variables, method = 'EF', verbose = False):
    
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

def normal_variable_iteratively_solving(mean, std, start, end, step, 
                                        options, model, scenarioData, scenarioProbability, firstStageObjective, firstStageVariables, 
                                        dimension_Out_of_Sample_evaluation,
                                        df = pd.DataFrame(columns=['n', 'RP', 'RP_Out_of_Sample', 'WS', 'EVPI', 'VSS']), method = 'EF', 
                                        scenarios = {} , seed1=None, seed2= None, lb = 0):
    if seed1 is not None:
        np.random.seed(seed1)
    
    Out_of_Sample_evaluation_Scenarios = {}
    Out_of_Sample_evaluation_Scenarios[1] = np.random.normal(mean, std, dimension_Out_of_Sample_evaluation)
    Out_of_Sample_evaluation_Scenarios[1] = [v for v in Out_of_Sample_evaluation_Scenarios[1] if v>0]
    
    n1 = len(Out_of_Sample_evaluation_Scenarios[1])
    Out_of_Sample_evaluation_Scenarios[2] = [3]*n1
    Out_of_Sample_evaluation_Scenarios[3] = [2]*n1

    scenarioProbabilities = [1/n1]*n1

    all_scenario_names_Out_of_Sample = [str(i) for i in range(int(n1))]

    scenario_creator_kwargs_Out_of_Sample={
                        'model': model, 
                        'scenarioData': scenarioData,
                        'scenarios': Out_of_Sample_evaluation_Scenarios, 
                        'scenariosProbabilities': scenarioProbabilities,
                        'scenarioProbability': scenarioProbability,
                        'firstStageObjective': firstStageObjective, 
                        'firstStageVariables': firstStageVariables}

    if seed2 is not None:
        np.random.seed(seed2)

    for n in range(start,end,step):
        scenarios[n] = {}
        
        if n == start:
            scenarios[n][1] = np.random.normal(mean, std, start+step)
        else:    
            scenarios[n][1] = np.append(scenarios[n-step][1], np.random.normal(mean, std, step)) 
        
        scenarios[n][1] = [v for v in scenarios[n][1] if v>0]

        n1 = len(scenarios[n][1])
        scenarios[n][2] = [3]*n1
        scenarios[n][3] = [2]*n1
        scenarioProbabilities = [1/n1]*n1

        all_scenario_names = [str(i) for i in range(int(n1))]
        scenario_creator_kwargs={
                        'model': model, 
                        'scenarioData': scenarioData,
                        'scenarios': scenarios[n], 
                        'scenariosProbabilities': scenarioProbabilities,
                        'scenarioProbability': scenarioProbability,
                        'firstStageObjective': firstStageObjective, 
                        'firstStageVariables': firstStageVariables}

        
        options_copy = copy.deepcopy(options)

        if method == 'LS' and 'valid_eta_lb' in options.keys():
            bounds = {name: lb for name in all_scenario_names}
            options_copy['valid_eta_lb'] = bounds


        RP = RP_solution(options_copy, all_scenario_names, scenario_creator_kwargs, method)
        WS = WS_solution(options_copy, all_scenario_names, scenario_creator_kwargs, method)['WS']

        if method == 'EF':
            RP_value = RP[0].get_objective_value()
            RP_Out_of_Sample = Out_of_sample_evaluation(options_copy, 
                                                        all_scenario_names_Out_of_Sample,
                                                        scenario_creator_kwargs_Out_of_Sample,
                                                        RP[0].ef.component_objects(pyo.Var),
                                                        method)
            
            if type(RP_Out_of_Sample) == str:
                RP_Out_of_Sample_value = RP_Out_of_Sample
            else:
                RP_Out_of_Sample_value = RP_Out_of_Sample[0].get_objective_value()
            
            EVPI = abs(WS-RP_value)
        elif method == 'LS':
            RP_value = RP[0].root.obj()
            RP_Out_of_Sample = Out_of_sample_evaluation(options_copy, 
                                                        all_scenario_names_Out_of_Sample,
                                                        scenario_creator_kwargs_Out_of_Sample,
                                                        RP[0].root.component_objects(pyo.Var),
                                                        method)

            if type(RP_Out_of_Sample) == str:
                RP_Out_of_Sample_value = RP_Out_of_Sample
            else:
                RP_Out_of_Sample_value = RP_Out_of_Sample[0].root.obj()

            EVPI = abs(WS-RP_value)
        
        VSS = VSS_value(options_copy, all_scenario_names, scenario_creator_kwargs, method, RP = RP)


        df = pd.concat([df, pd.DataFrame({'n':[n], 'RP': [RP_value], 'RP_Out_of_Sample' : [RP_Out_of_Sample_value], 'WS': [WS], 'EVPI':[EVPI], 'VSS':[VSS]})], ignore_index=True)
    return df, scenarios