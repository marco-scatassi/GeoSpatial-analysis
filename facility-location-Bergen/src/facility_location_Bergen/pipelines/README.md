About this folder
==============================
It contains all the pipelines used in the project. In particular:

1. The pipelines are used to run the different scripts in the project. 
2. The pipelines are written in Python and are run from the command line. 
3. The pipelines are run from the root folder of the project.

The pipeline developed so far, are the following:

1. `ingestion`: it ingests the data from the source and stores it in a **MongoDB** collection.
2. `cleaning`: it cleans the data and stores it in a **MongoDB** collection.
3. `convert_to_gdf`: it converts the data from a **MongoDB** collection to a **GeoPandas** dataframe and stores it in the `data\03_primary` folder.
4. `create_average_gdfs`: it creates the average **GeoPandas** dataframes and stores them in the `data\03_primary` folder.
5. `build_average_graphs`: it builds the average graphs, starting from the average gdfs, and stores them in the `data\03_primary` folder.
6. `build_adjacency_matrix`: it builds the adjacency matrix, starting from the average graphs, and stores it in the `data\03_primary` folder.
7. `problem_solution`: it solves the facility location problem and stores the results in the `data\07_model_output` folder.



> **WARNING**
> 
> the **problem_solution** pipeline is not a real pipeline. It is simple a script inside the `src\facility_location_Bergen\pipelines\problem_solution` folder. It has been added here for choerence with the rest of the project.