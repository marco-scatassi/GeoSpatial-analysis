from shapely.geometry import MultiLineString, MultiPoint, LineString


def toMultiLineString(geom):
    return MultiLineString([LineString(e["coordinates"]) for e in geom])


def toExtremePoints(geom):
    if type(geom) == MultiLineString:
        c0 = geom.geoms[0].coords[0]
        c1 = geom.geoms[-1].coords[-1]
    elif type(geom) == LineString:
        c0 = geom.coords[0]
        c1 = geom.coords[-1]

    return MultiPoint([c0, c1])
