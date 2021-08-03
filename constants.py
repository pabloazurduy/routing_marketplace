BETA_INIT = {'ft_has_geo_0': 0.0,
             'ft_has_geo_1': 0.0,
             'ft_has_geo_10': -1.548509445677352,
             'ft_has_geo_11': 5891.173533038039,
             'ft_has_geo_12': 6.604406027913742e-10,
             'ft_has_geo_13': 3.0595409247038517e-14,
             'ft_has_geo_14': 497.96718947271785,
             'ft_has_geo_15': 0.0,
             'ft_has_geo_16': 0.0,
             'ft_has_geo_17': 1058.8369762434686,
             'ft_has_geo_18': -1020.8919056679359,
             'ft_has_geo_19': 0.0,
             'ft_has_geo_2': -710.3518783815902,
             'ft_has_geo_20': 813.2848446296128,
             'ft_has_geo_21': -176.74050041491589,
             'ft_has_geo_22': 464.13300284658885,
             'ft_has_geo_23': 0.0,
             'ft_has_geo_24': 4642.7392195890625,
             'ft_has_geo_25': 30.545219210024282,
             'ft_has_geo_26': 0.4019107792898208,
             'ft_has_geo_27': -61.65359651109964,
             'ft_has_geo_28': 147.6413115624648,
             'ft_has_geo_29': -0.0,
             'ft_has_geo_3': 6.845232794010831,
             'ft_has_geo_30': 3094.6866415135514,
             'ft_has_geo_31': 1015.3138075909125,
             'ft_has_geo_32': -683.9412117478197,
             'ft_has_geo_33': 677.7650607087504,
             'ft_has_geo_34': -829.8463346908943,
             'ft_has_geo_35': 0.0,
             'ft_has_geo_36': 723.0250040801059,
             'ft_has_geo_37': -521.012154669868,
             'ft_has_geo_38': 188.44366252967077,
             'ft_has_geo_39': 31.85976274862604,
             'ft_has_geo_4': -602.4383262555745,
             'ft_has_geo_40': -599.8865036373118,
             'ft_has_geo_41': -980.7645124571914,
             'ft_has_geo_42': 0.0,
             'ft_has_geo_43': -19.82505834702592,
             'ft_has_geo_44': 205.10490180457356,
             'ft_has_geo_45': -943.6079761068675,
             'ft_has_geo_46': 0.0,
             'ft_has_geo_47': -145.47373108291666,
             'ft_has_geo_48': -69.26846443114253,
             'ft_has_geo_49': -1101.8862234606154,
             'ft_has_geo_5': -0.0,
             'ft_has_geo_50': 197.24566291552134,
             'ft_has_geo_51': -323.11692797374826,
             'ft_has_geo_6': 122.95021590648132,
             'ft_has_geo_7': -4427.857053386289,
             'ft_has_geo_8': -1116.6389381100166,
             'ft_has_geo_9': -2583.5388265377965,
             'ft_inter_geo_dist': -2497.874346109264,
             'ft_size':           0.6000352603169622,
             'ft_size_drops':     0.0,
             'ft_size_geo':     -1.8563571979088904, # 956.4919392614147,
             'ft_size_pickups': -2.992447600045428
            }

BETA_TEST = {'ft_has_geo_10':-1.3575579066633872,
              'ft_has_geo_11':-0.0,
              'ft_has_geo_12':-0.0,
              'ft_has_geo_13':-0.0,
              'ft_has_geo_17':7.167277585334053,
              'ft_has_geo_18':-26.91088655468027,
              'ft_has_geo_2':21.175125607799274,
              'ft_has_geo_20':0.005747803893933877,
              'ft_has_geo_21':5.407439202291939e-15,
              'ft_has_geo_27':-0.0,
              'ft_has_geo_29':0.2398790059892945,
              'ft_has_geo_3':17.86569412253493,
              'ft_has_geo_30':-0.2620162785001782,
              'ft_has_geo_31':0.21765885480990793,
              'ft_has_geo_32':0.0,
              'ft_has_geo_34':18.929455585541795,
              'ft_has_geo_36':-0.0,
              'ft_has_geo_37':1.3401044979593064e-14,
              'ft_has_geo_38':-0.0,
              'ft_has_geo_39':-0.03880601083088704,
              'ft_has_geo_40':-11.915590900105615,
              'ft_has_geo_42':-0.0,
              'ft_has_geo_43':-1.4288233673253905,
              'ft_has_geo_45':-50.2448538791531,
              'ft_has_geo_46':0.0,
              'ft_has_geo_47':3.035072193569022e-14,
              'ft_has_geo_48':41.917014602323384,
              'ft_has_geo_49':0.0,
              'ft_has_geo_5':-0.09153453599903799,
              'ft_has_geo_50':-0.0,
              'ft_has_geo_51':0.0,
              'ft_has_geo_6':-0.0,
              'ft_has_geo_7':-8.493206138382448e-15,
              'ft_has_geo_8':-0.0,
              'ft_has_geo_9':49.01021072700433,
              'ft_inter_geo_dist':-10.356305229453124,
              'ft_size':       -1.5950020414588337,
              'ft_size_drops': -1.192355240550351,
              'ft_size_geo':   -1.8563571979088904,
              'ft_size_pickups':11.095607692852028,
            }

ROUTE_FEATURES = list(BETA_INIT.keys())

FEATURES_STAT_SAMPLE = {
    'linear_gral_features_util':   {'weight':0.20,'mean':-1732.120277,'std':1911.422243},
    'linear_has_geo_features_util':{'weight':0.10,'mean':-403.2038632,'std':1216.913311},
    'price_util':                  {'weight':0.30,'mean':4017.627266, 'std':2278.858993},
    'origin_util':                 {'weight':0.30,'mean':-1501.906135,'std':1363.956983},
    'route_len_util':              {'weight':0.10,'mean':260.1442661, 'std':471.5637154},
}

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