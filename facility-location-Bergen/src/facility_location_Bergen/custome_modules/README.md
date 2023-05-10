
# Documentation for package p-center facility-location-Bergen

## Class AdjacencyMatrix
This class is designed to create an adjacency matrix from a list of coordinates. 

### Class attributes
- **coordinates**: 
  - a gpd.geoseries.GeoSeries object containing the input coordinates that are used to create the adjacency matrix.
- **mode**: 
  - a string indicating the type of travel mode used to calculate the distances between the coordinates when kind is set to "osrm". The default value is "distance".
- **kind**: 
  - a string indicating the type of adjacency matrix to create. The options are "geometric" or "osrm". The default value is "geometric".
- **epsg**: 
  - an integer indicating the coordinate reference system (CRS) code used for the input coordinates. The default value is 32610, which corresponds to the WGS 84 / UTM zone 10N CRS.
- **adjacency_matrix**: 
  - a NumPy array containing the adjacency matrix that is created from the input coordinates. The values in the array represent the distances between pairs of points in the coordinate set. The dimensions of the array are n by n, where n is the number of input coordinates. If kind is set to "geometric", the distances are calculated using the Euclidean distance between the points. If kind is set to "osrm", the distances are calculated using the OpenStreetMap Routing Machine (OSRM) API.

### Class public methods
  
- `__init__(self, coordinates: gpd.geoseries.GeoSeries, kind="geometric", epsg=32610, mode="distance")`:
  - This method is used to initialize the class.


- `create_adjacency_matrix(self)`:
  - This method is used to create an **adjacency matrix** for a set of coordinates representing geographic points. The method takes into account three different cases based on the value of the attributes **kind** and **mode**:

  - "**Geometric**" Case:
    - If `self.kind` is set to "**geometric**", the method first converts the coordinates to a specific coordinate reference system (CRS) using the to_crs() method from a geopandas (gpd) GeoSeries object. Then, it calculates the pairwise distances between the coordinates. The calculated distances are stored in a **square matrix**, which represents the adjacency matrix, where n is the number of coordinates.

  - "**OSRM**" Case:
    - If `self.kind` is set to "**osrm**", the method assumes that the coordinates are in a format that can be used with the **OpenStreetMap Routing Machine (OSRM) API**. It first prepares the coordinates by converting them into a list of coordinate pairs. Then, it constructs URLs for making API calls to the OSRM service in chunks of 100 coordinates at a time (to avoid exceeding API limitations). For each pair of source and destination chunks, it makes an API request to OSRM to obtain the distances or durations between all pairs of coordinates in those chunks (depending on the value of `self.mode`), using the requests library. The resulting distances are extracted from the JSON response, and they are stored in the **distances numpy array**, which represents the adjacency matrix.

- `save(self, path: str)`:
  - This method is used to save the adjacency matrix to a file. 
    - It takes as argument a string representing the path to the file as input. 
    - The method saves the adjacency matrix to a file in the **.pkl** format.

- `load(self, path: str)`:
  - This method is used to load an adjacency matrix from a file. 
    - It takes as argument a string representing the path to the file as input. 
    - The method loads the adjacency matrix from a file in the **.pkl** format.

## Class FacilityLocation 
This class is designed to implement two algorithms for the uncapacitated facility location problem. 
- The first algorithm is a greedy approach (GON). 
- The second algorithm is an exact algorithm based on the integer linear programming (ILP) model.

### Class attributes
- **coordinates**: 
  - a GeoSeries containing the coordinates of the demand points.
- **n_of_locations_to_choose**: 
  - the number of facilities to open.
- **adjacency_matrix**: 
  - a NumPy array representing the adjacency matrix between demand points.
- **adjacency_matrix_weight**: 
  - a string representing the type of input matrix.
- **n_of_demand_points**: 
  - the number of demand points.
- **solution_value**: 
  - the objective value of the best solution found by the algorithm.
- **algorithm**: 
  - a string representing the algorithm used to solve the problem (either "GON" or "exact").
- **solver_status**: 
  - the status of the solver.
- **instance**: 
  - the instance of the Pyomo model.
- **result**: 
  - the result of solving the Pyomo model.
- **computation_time**: 
  - the time taken to solve the problem.

### Class public methods

- `__init__(self, coordinates: gpd.geoseries.GeoSeries, n_of_locations_to_choose: int, adjacency_matrix: AdjacencyMatrix)`:
  - This method is used to initialize the class. It takes in four arguments:
    - **coordinates**: 
      - a GeoSeries containing the coordinates of the demand points.
    - **n_of_locations_to_choose**: 
      - the number of facilities to open.
    - **adjacency_matrix**: 
      - an AdjacencyMatrix object representing the adjacency matrix between demand points.

- `solve(self, mode = "exact", algorithm = "gon", n_trial = None)`: 
  - It is used to solve the facility location problem. It takes in three arguments:
    - **mode**: 
      - This argument specifies whether the solution should be exact or approximate. It has two possible values: "**exact**" or "**approx**".
    - **algorithm**: 
      - This argument is used when mode is set to "**approx**". It specifies the algorithm to be used for the approximate solution. It has two possible values: "**gon**" or "**gon_plus**".
    - **n_trial**: 
      - This argument is used when algorithm is set to "**gon_plus**". It specifies the number of trials to be used in the **GON+** algorithm.

- `plot(self)` this method plots the solution computed by the solve() method. 
  - If the locations_coordinates attribute is **None**, the method returns the message "**solve the problem first**". 
  - Otherwise, the method creates a scatter plot with **blue** markers for the coordinates of the **original** **points** and **red** **markers** for the **coordinates** of the **points** **selected** as locations. The method returns the fig and ax objects of the plot.

- `save(self, path: str)`:
  - This method is used to save the solution to a file. 
    - It takes as argument a string representing the path to the file as input. 
    - The method saves the solution to a file in the **.pkl** format.

- `load(self, path: str)`:
  - This method is used to load a solution from a file. 
    - It takes as argument a string representing the path to the file as input. 
    - The method loads the solution from a file in the **.pkl** format.

## class FacilityLocationReport
The **FacilityLocationReport** class is used to represent a report containing facility location information. 

### Class attributes
- **facility_locations**: 
  - a list of FacilityLocation objects, which represents the facility location solutions.

### Class public methods
The class has three main methods:

- `__init__(self, facility_locations: list[FacilityLocation])`: 
  - This is the constructor method for the **FacilityLocationReport** class. It takes a list of **FacilityLocation** objects as input, which represents the facilities and their locations. It initializes the object with the provided list of FacilityLocation objects.

- `graphical_algorithm_solutions_comparison(self)`: 
  - This method generates **scatter plots** to compare facility location solutions for different algorithms. It uses the matplotlib library to create scatter plots, with one plot for each **algorithm** and **weight** combination. This can help visually analyze and compare the performance of different algorithms in solving facility location problems.

- `graphical_adjacency_matrix_solutions_comparison(self)`: 
  - This method generates scatter plots to compare facility location solutions for different **algorithms**, with different colors for each **weight** combination to avoid overlapping of points. It uses the matplotlib library to create scatter plots, with one plot for each algorithm. This can help visually analyze and compare the performance of different algorithms in solving facility location problems, while taking into account different weight combinations.