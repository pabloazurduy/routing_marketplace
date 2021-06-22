KEPLER_CONFIG = {
  "version": "v1",
  "config": {
    "visState": {
      "filters": [
        {
          "dataId": [
            "cluster"
          ],
          "id": "3sfs2aq97",
          "name": [
            "node_type"
          ],
          "type": "multiSelect",
          "value": [],
          "enlarged": False,
          "plotType": "histogram",
          "animationWindow": "free",
          "yAxis": None
        },
        {
          "dataId": [
            "cluster"
          ],
          "id": "qetpbicvs",
          "name": [
            "cluster"
          ],
          "type": "range",
          "value": [
            0,
            24
          ],
          "enlarged": False,
          "plotType": "histogram",
          "animationWindow": "free",
          "yAxis": None
        }
      ],
      "layers": [
        {
          "id": "bcw7eaq",
          "type": "geojson",
          "config": {
            "dataId": "geos",
            "label": "geos",
            "color": [
              255,
              254,
              230
            ],
            "columns": {
              "geojson": "geometry"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.01,
              "strokeOpacity": 0.8,
              "thickness": 0.2,
              "strokeColor": [
                221,
                178,
                124
              ],
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "strokeColorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "radius": 10,
              "sizeRange": [
                0,
                10
              ],
              "radiusRange": [
                0,
                50
              ],
              "heightRange": [
                0,
                500
              ],
              "elevationScale": 5,
              "stroked": True,
              "filled": False,
              "enable3d": False,
              "wireframe": False
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "strokeColorField": None,
            "strokeColorScale": "quantile",
            "heightField": None,
            "heightScale": "linear",
            "radiusField": None,
            "radiusScale": "linear"
          }
        },
        {
          "id": "ugn6rg2",
          "type": "point",
          "config": {
            "dataId": "cluster",
            "label": "Point",
            "color": [
              136,
              87,
              44
            ],
            "columns": {
              "lat": "lat",
              "lng": "lng",
              "altitude": None
            },
            "isVisible": True,
            "visConfig": {
              "radius": 10,
              "fixedRadius": False,
              "opacity": 0.8,
              "outline": False,
              "thickness": 2,
              "strokeColor": None,
              "colorRange": {
                "name": "ColorBrewer Set2-6",
                "type": "qualitative",
                "category": "ColorBrewer",
                "colors": [
                  "#66c2a5",
                  "#fc8d62",
                  "#8da0cb",
                  "#e78ac3",
                  "#a6d854",
                  "#ffd92f"
                ]
              },
              "strokeColorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "radiusRange": [
                0,
                50
              ],
              "filled": True
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": {
              "name": "cluster",
              "type": "integer"
            },
            "colorScale": "quantile",
            "strokeColorField": None,
            "strokeColorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear"
          }
        },
        {
          "id": "831siew",
          "type": "geojson",
          "config": {
            "dataId": "inter_geo",
            "label": "inter_geo",
            "color": [
              255,
              153,
              31
            ],
            "columns": {
              "geojson": "shape"
            },
            "isVisible": False,
            "visConfig": {
              "opacity": 0.8,
              "strokeOpacity": 0.8,
              "thickness": 0.5,
              "strokeColor": None,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "strokeColorRange": {
                "name": "ColorBrewer Set2-6",
                "type": "qualitative",
                "category": "ColorBrewer",
                "colors": [
                  "#66c2a5",
                  "#fc8d62",
                  "#8da0cb",
                  "#e78ac3",
                  "#a6d854",
                  "#ffd92f"
                ]
              },
              "radius": 10,
              "sizeRange": [
                0,
                10
              ],
              "radiusRange": [
                0,
                50
              ],
              "heightRange": [
                0,
                500
              ],
              "elevationScale": 5,
              "stroked": True,
              "filled": False,
              "enable3d": False,
              "wireframe": False
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "strokeColorField": {
              "name": "cluster",
              "type": "integer"
            },
            "strokeColorScale": "quantile",
            "heightField": None,
            "heightScale": "linear",
            "radiusField": None,
            "radiusScale": "linear"
          }
        }
      ],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "geos": [
              {
                "name": "id",
                "format": None
              },
              {
                "name": "name",
                "format": None
              },
              {
                "name": "area",
                "format": None
              }
            ],
            "cluster": [
              {
                "name": "node_sid",
                "format": None
              },
              {
                "name": "node_type",
                "format": None
              },
              {
                "name": "node_id",
                "format": None
              },
              {
                "name": "cluster",
                "format": None
              },
              {
                "name": "geo_id",
                "format": None
              }
            ],
            "inter_geo": [
              {
                "name": "cluster",
                "format": None
              },
              {
                "name": "geo_i",
                "format": None
              },
              {
                "name": "geo_j",
                "format": None
              }
            ]
          },
          "compareMode": False,
          "compareType": "absolute",
          "enabled": True
        },
        "brush": {
          "size": 0.5,
          "enabled": False
        },
        "geocoder": {
          "enabled": False
        },
        "coordinate": {
          "enabled": False
        }
      },
      "layerBlending": "normal",
      "splitMaps": [],
      "animationConfig": {
        "currentTime": None,
        "speed": 1
      }
    },
    "mapState": {
      "bearing": 0,
      "dragRotate": False,
      "latitude": -33.49753419273629,
      "longitude": -70.77098329764567,
      "pitch": 0,
      "zoom": 9.973998208213096,
      "isSplit": False
    },
    "mapStyle": {
      "styleType": "dark",
      "topLayerGroups": {},
      "visibleLayerGroups": {
        "label": True,
        "road": True,
        "border": False,
        "building": True,
        "water": True,
        "land": True,
        "3d building": False
      },
      "threeDBuildingColor": [
        9.665468314072013,
        17.18305478057247,
        31.1442867897876
      ],
      "mapStyles": {}
    }
  }
}