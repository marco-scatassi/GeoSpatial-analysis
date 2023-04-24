import shapely as shp

def toMultiLineString(geom):
    return shp.multilinestrings(
        [shp.linestrings(e['coordinates']) for e in geom])
    
def toExtremePoints(geom: shp.MultiLineString):
    c0 = geom.geoms[0].coords[0]
    c1 = geom.geoms[-1].coords[-1]
    
    return shp.multipoints([c0, c1])