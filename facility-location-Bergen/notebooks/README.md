# Observations and results
## 3_clening.ipynb
### Observation 1
Data available within a day **differ** at temporal levels, therefor, given a road segment, the number of observations available for a given day is not the same for all temporal levels.

### Observation 2
Actually, in this notebook, we clean the data by removing the observations that are not available at all temporal levels. This is **not** necessary **the best solution**, expecially when the goal is to **aggregate** data at a higher temporal level

### TODO
- [x] **Modify** the data cleaning pipeline in such a way that only the spatial level is considered
- [x] **Replace** cleaning collections with the ones obtained by the new pipeline
- [x] **Replace** gdf with the one obtained by running again the convert_to_gdf pipeline

### Note
All the changes won't be applied in the notebook, but in the package, since the notebook is used only as playground, it is not necessary to keep it up to date with the package.

## 5.1_average_graph_analysis.ipynb
### Observation 1
The **graph**, obtained considering data in a specific day common to all times, is **not connected**. This means that there are some **road segments** that are **not connected** to the rest of the graph. 

### Observation 2
Many **extreme points** of disconnected road segments are **very close to each other** (for example because they belong to the same roundabout). So, it should be possible to merge those points in a single one in order to reduce the graph's disconnected components. 

### TODO 
- [ ] **Find** a way to merge extreme points of disconnected road segments
