import sys

sys.path.append(
    r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
)

import dill
import json
import time
import requests
import numpy as np
import geopandas as gpd
from pyomo.environ import *
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from log import print_INFO_message_timestamp, print_INFO_message


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


# class to define the Facility Location problem
class FacilityLocation:

    # ----------------------------------------------- define the constructor -----------------------------------------------

    locations_coordinates = None
    locations_index = None
    solution_value = None
    algorithm = None
    solver_status = None
    model = None
    instance = None
    result = None
    computation_time = None

    def __init__(
        self,
        coordinates: gpd.geoseries.GeoSeries,
        n_of_locations_to_choose: int,
        adjancency_matrix: AdjacencyMatrix,
        candidate_coordinates: gpd.geoseries.GeoSeries = None,
    ):

        self.coordinates = coordinates
        if candidate_coordinates is None:
            self.candidate_coordinates = coordinates
        else:
            self.candidate_coordinates = candidate_coordinates
        self.n_of_locations_to_choose = n_of_locations_to_choose
        self.adjacency_matrix = adjancency_matrix.adjacency_matrix
        self.adjacency_matrix_weight = self.__get_weight(adjancency_matrix)
        self.n_of_demand_points = len(coordinates)

    # --------------------------- define the private method to retrieve the type of input matrix ---------------------------
    def __get_weight(self, adjancency_matrix):
        if adjancency_matrix.kind == "geometric":
            return " ".join([adjancency_matrix.kind, "distance"])
        else:
            return " ".join([adjancency_matrix.kind, adjancency_matrix.mode])

    # --------------------------------------------- implement the GON algorithm --------------------------------------------
    def __max_dist_point(self, C):

        min_point_dist = self.adjacency_matrix[C, :].min(axis=0)
        max_index = min_point_dist.argmax()

        return max_index

    # a single iteration of the GON algorithm
    def __global_iteration(self, C):
        for i in range(self.n_of_locations_to_choose - 1):
            # find the point that is the farthest from the set C
            # print time for each iteration to see the progress of the algorithm
            max_index = self.__max_dist_point(C)
            C.append(max_index)

        max_index = self.__max_dist_point(C)

        return self.adjacency_matrix[C, max_index].min()

    # main function
    def __solve_gon(self):
        np.random.seed(1783297)

        C = []

        # choose a random point from V as the first centroid
        init = np.random.randint(0, self.n_of_demand_points)
        C.append(init)

        self.solution_value = self.__global_iteration(C)

        self.locations_index = C
        self.locations_coordinates = [
            self.coordinates.iloc[i] for i in self.locations_index
        ]

    def __solve_gon_plus(self, n_trial):
        if n_trial > 1 and n_trial <= self.n_of_demand_points:
            C_l = []
            d = []

            # choose n_trial random points from V as the first centroids
            init = np.random.randint(0, len(self.coordinates), n_trial)

            for i in range(n_trial):
                C_l.append([])
                C_l[i].append(init[i])

                d.append(self.__global_iteration(C_l[i]))

            self.solution_value = np.min(d)
            self.locations_index = C_l[np.argmin(d)]
            self.locations_coordinates = [
                self.coordinates.iloc[i] for i in self.locations_index
            ]

        else:
            return (
                "n_trial must be greater than 1 and less than the number of coordinates"
            )

    # --------------------------------------------- implement the exact algorithm ------------------------------------------
    # define model constraints
    def __completeSingleCoverage(self, model, i):
        return sum(model.y[i, j] for j in model.J) == 1

    def __maximumLocations(self, model):
        return sum([model.x[j] for j in model.J]) == pyo.value(model.p)

    def __maximalDistance(self, model, i):
        return sum(model.d[j, i] * model.y[i, j] for j in model.J) <= model.L

    def __servedByOpenFacility(self, model, i, j):
        return model.y[i, j] <= model.x[j]

    # define the objective function
    def __maximalDistanceObj(self, model):
        return model.L

    def __DefineAbstractModel(self):
        # -------------------------abastract model----------------------------
        model = pyo.AbstractModel()

        # ---------------------------index sets-------------------------------
        model.I = pyo.Set(initialize=list(self.coordinates.index))
        model.J = pyo.Set(initialize=list(self.candidate_coordinates.index))

        # ---------------------------parameters-------------------------------
        # define the number of locations to be opened (p)
        model.p = pyo.Param(within=PositiveIntegers)

        # define the distance matrix (d)
        model.d = pyo.Param(model.J, model.I, within=NonNegativeReals)

        # ---------------------------variables--------------------------------
        # define the binary variables for the location decision (x)
        model.x = Var(model.J, within=Binary)

        # define the binary variables for the assignment decision (y)
        model.y = Var(model.I, model.J, within=Binary)

        # define the auxiliary variable for the maximal distance (L)
        model.L = Var(within=NonNegativeReals)

        # --------------------------constraints-------------------------------
        # define a constraint for each demand point to be covered by a single location
        model.completeSingleCoverage = Constraint(
            model.I, rule=self.__completeSingleCoverage
        )

        # define a constraint for the maximum number of locations
        model.maximumLocations = Constraint(rule=self.__maximumLocations)

        # define a constraint for the maximal distance (L is an auxiliary variable)
        model.maximalDistance = Constraint(model.I, rule=self.__maximalDistance)

        # define a constraint for each demand point to be served by an open facility
        model.servedByOpenFacility = Constraint(
            model.I, model.J, rule=self.__servedByOpenFacility
        )

        # -----------------------objective function---------------------------
        model.maximalDistanceObj = Objective(
            rule=self.__maximalDistanceObj, sense=minimize
        )

        self.model = model

        return model

    def __solve_exact(self):
        print_INFO_message("Defining the abstract model...")
        model = self.__DefineAbstractModel()

        print_INFO_message_timestamp("Initializing data...")
        distance_data = {
            (j, i): self.adjacency_matrix[j][i]
            for j in self.candidate_coordinates.index
            for i in self.coordinates.index
        }
        data = {None: {"p": {None: self.n_of_locations_to_choose}, "d": distance_data}}

        print_INFO_message("Creating the instance...")
        self.instance = model.create_instance(data)

        print_INFO_message_timestamp("Solving the model...")
        opt = SolverFactory("cplex")

        self.result = opt.solve(self.instance)

        self.solution_value = self.instance.L.value
        self.locations_index = [
            j for j in self.candidate_coordinates.index if self.instance.x[j].value == 1
        ]
        self.locations_coordinates = [
            self.candidate_coordinates.loc[j] for j in self.locations_index
        ]

        # Update the status of the solver
        if (
            self.result.solver.status == pyo.SolverStatus.ok
            and self.result.solver.termination_condition
            == pyo.TerminationCondition.optimal
        ):
            self.solver_status = "Optimal solution found"
        elif (
            self.result.solver.termination_condition
            == pyo.TerminationCondition.infeasible
        ):
            self.solver_status = "Problem is infeasible"
        else:
            self.solver_status = self.result.solver.termination_condition

    # ---------------------------------------- implement the methods to solve the problem -----------------------------------
    def solve(self, mode="exact", algorithm="gon", n_trial=None):
        t1 = time.time()

        if mode == "exact":
            print_INFO_message_timestamp("Solving the problem exactly...")
            self.__solve_exact()

        elif mode == "approx":
            if algorithm == "gon":
                print_INFO_message_timestamp(
                    "Solving the problem approximately using the GON algorithm..."
                )
                self.__solve_gon()

            elif algorithm == "gon_plus":
                print_INFO_message_timestamp(
                    "Solving the problem approximately using the GON+ algorithm..."
                )
                self.__solve_gon_plus(n_trial)

        else:
            return "mode must be either 'exact' or 'approx'"

        t2 = time.time()

        self.computation_time = t2 - t1
        self.algorithm = mode if mode == "exact" else algorithm

    # ---------------------------------------- implement the methods to visualize the solution -----------------------------------
    def plot(self):
        if self.locations_coordinates == None:
            return "solve the problem first"

        fig, ax = plt.subplots()

        ax.scatter(
            [c.x for c in self.coordinates.geometry],
            [c.y for c in self.coordinates.geometry],
            c="blue",
        )
        ax.scatter(
            [c[0].x for c in self.locations_coordinates],
            [c[0].y for c in self.locations_coordinates],
            c="red",
        )

        return fig, ax

    # ---------------------------------------- implement the methods to save and load the matrix -----------------------------------
    # save the solution
    def save(self, file_name):
        if self.locations_coordinates == None:
            return "solve the problem first"

        with open(file_name, "wb") as f:
            dill.dump(self, f)

    # load the solution
    @staticmethod
    def load(file_name):
        with open(file_name, "rb") as f:
            a = dill.load(f)

        return a


# class to visualize fFacility location problem solutions
class FacilityLocationReport:

    # ---------------------------------------------------------------- define the constructor ---------------------------------------------------------------
    def __init__(self, facility_locations: dict[FacilityLocation]):
        self.facility_locations = list(facility_locations.values())
        self.keys = list(facility_locations.keys())

    # ---------------------------------------------------------------- define the methods -------------------------------------------------------------------
    # function to jitter the data to avoid overlapping of the points
    def __rand_jitter(self, list):
        scale_factor = max(list) - min(list)
        if scale_factor != 0:
            stdev = 0.01 * (scale_factor)
        else:
            stdev = 0.0008
        return list + np.random.randn(len(list)) * stdev

    # get the number of different algorithms
    def __adjacency_matrix_weight_from_list_to_dict(self):
        d = {}
        n = len(self.facility_locations)

        for i in range(n):
            if self.facility_locations[i].adjacency_matrix_weight not in d.keys():
                d[self.facility_locations[i].adjacency_matrix_weight] = [i]
            else:
                d[self.facility_locations[i].adjacency_matrix_weight].append(i)
        return d

    def __algorithm_from_list_to_dict(self):
        d = {}
        n = len(self.facility_locations)

        for i in range(n):
            if self.facility_locations[i].algorithm not in d.keys():
                d[self.facility_locations[i].algorithm] = [i]
            else:
                d[self.facility_locations[i].algorithm].append(i)
        return d

    # plot scatter plots of the facility location solutions, one for each algorithm
    def graphical_algorithm_solutions_comparison(self):

        weight_names_and_index = self.__adjacency_matrix_weight_from_list_to_dict()
        algorithm_names_and_index = self.__algorithm_from_list_to_dict()

        n = len(weight_names_and_index)
        m = len(algorithm_names_and_index)

        fig, axs = plt.subplots(
            nrows=n, ncols=m, sharex=True, sharey=True, figsize=(15, 15)
        )

        for i, algorithm_name in enumerate(algorithm_names_and_index.keys()):
            for j, weight_name in enumerate(weight_names_and_index.keys()):
                for k in weight_names_and_index[weight_name]:
                    if self.facility_locations[k].algorithm == algorithm_name:
                        if weight_name[-8::1] == "duration":
                            unit = "s"
                        else:
                            unit = "m"

                        axs[j, i].scatter(
                            [
                                x
                                for x in np.array(
                                    self.facility_locations[k].coordinates
                                )[:, 0]
                            ],
                            [
                                x
                                for x in np.array(
                                    self.facility_locations[k].coordinates
                                )[:, 1]
                            ],
                            c="blue",
                        )
                        axs[j, i].scatter(
                            [
                                x
                                for x in np.array(
                                    self.facility_locations[k].locations_coordinates
                                )[:, 0]
                            ],
                            [
                                x
                                for x in np.array(
                                    self.facility_locations[k].locations_coordinates
                                )[:, 1]
                            ],
                            c="red",
                        )
                        axs[j, i].text(
                            0.05,
                            0.2,
                            f"points: {self.facility_locations[k].n_of_demand_points} \ntime: {round(self.facility_locations[k].computation_time,3)} s\nsolution: {round(self.facility_locations[k].solution_value, 2)} "
                            + unit,
                            transform=axs[j, i].transAxes,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
                        )
                        axs[j, i].set_title(self.facility_locations[k].algorithm)
                        axs[j, i].set_xlabel("Longitude")
                        axs[j, i].set_ylabel("Latitude")
                        axs[j, i].set_title(
                            f"{self.facility_locations[k].algorithm} - {self.facility_locations[k].adjacency_matrix_weight}"
                        )

    def graphical_adjacency_marix_solutions_comparison(self):
        colors = ["red", "green", "orange", "pink", "brown", "grey", "olive", "cyan"]

        algorithm_names_and_index = self.__algorithm_from_list_to_dict()
        n = len(algorithm_names_and_index.keys())

        if n > 1:
            fig, axs = plt.subplots(nrows=1, ncols=n, sharey=True, figsize=(15, 5))
        else:
            fig, axs = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(5, 5))
            axs = [axs]

        for i, algorithm_name in enumerate(algorithm_names_and_index.keys()):
            axs[i].scatter(
                [c.x for c in self.facility_locations[i].coordinates.geometry],
                [c.y for c in self.facility_locations[i].coordinates.geometry],
                c="blue",
                alpha=0.5,
            )
            axs[i].set_title(algorithm_name)
            axs[i].set_xlabel("Longitude")
            axs[i].set_ylabel("Latitude")

            for k, j in enumerate(algorithm_names_and_index[algorithm_name]):
                x = self.__rand_jitter(
                    [
                        c["geometry"].coords.xy[0][0]
                        for c in self.facility_locations[j].locations_coordinates
                    ]
                )
                y = self.__rand_jitter(
                    [
                        c["geometry"].coords.xy[1][0]
                        for c in self.facility_locations[j].locations_coordinates
                    ]
                )
                axs[i].scatter(
                    x,
                    y,
                    c=colors[k],
                    label=self.facility_locations[j].adjacency_matrix_weight,
                )

            axs[i].legend()

    def graphical_keys_solutions_comparison(self):
        colors = ["red", "lawngreen", "magenta", "blue", "aqua", "darkorange"]
        algorithm_names_and_index = self.__algorithm_from_list_to_dict()
        n = len(algorithm_names_and_index.keys())

        if n > 1:
            fig, axs = plt.subplots(nrows=1, ncols=n, sharey=True, figsize=(15, 8))
        else:
            fig, axs = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(8, 10))
            axs = [axs]

        for i, algorithm_name in enumerate(algorithm_names_and_index.keys()):
            axs[i].scatter(
                [c.x for c in self.facility_locations[i].coordinates.geometry],
                [c.y for c in self.facility_locations[i].coordinates.geometry],
                c="grey",
                alpha=0.1,
            )
            axs[i].set_title(f"{algorithm_name} Solution locations")
            axs[i].set_xlabel("Longitude")
            axs[i].set_ylabel("Latitude")

            for k, j in enumerate(algorithm_names_and_index[algorithm_name]):
                x = self.__rand_jitter(
                    [
                        c["geometry"].coords.xy[0][0]
                        for c in self.facility_locations[j].locations_coordinates
                    ]
                )
                y = self.__rand_jitter(
                    [
                        c["geometry"].coords.xy[1][0]
                        for c in self.facility_locations[j].locations_coordinates
                    ]
                )
                axs[i].scatter(x, y, c=colors[k], label=self.keys[j])

            axs[i].legend()

            return fig