"""
Ontario Public Health Units — metadata, coordinates, population.
Source: Ontario Ministry of Health PHU boundary data.
"""

PHUS = [
    {"id": "TPH",  "name": "Toronto Public Health",                   "lat": 43.70, "lng": -79.42, "population": 2794356, "region": "Toronto"},
    {"id": "OPH",  "name": "Ottawa Public Health",                    "lat": 45.42, "lng": -75.69, "population": 1017449, "region": "East"},
    {"id": "HPH",  "name": "Hamilton Public Health",                  "lat": 43.25, "lng": -79.87, "population": 579200,  "region": "West"},
    {"id": "PPH",  "name": "Peel Public Health",                      "lat": 43.73, "lng": -79.72, "population": 1451820, "region": "Central"},
    {"id": "YPH",  "name": "York Region Public Health",               "lat": 44.00, "lng": -79.46, "population": 1109909, "region": "Central"},
    {"id": "DPH",  "name": "Durham Region Health",                    "lat": 43.90, "lng": -78.85, "population": 703971,  "region": "Central"},
    {"id": "HPPH", "name": "Halton Region Public Health",             "lat": 43.51, "lng": -79.88, "population": 595200,  "region": "Central"},
    {"id": "WRPH", "name": "Waterloo Region Public Health",           "lat": 43.45, "lng": -80.49, "population": 625600,  "region": "West"},
    {"id": "WEPH", "name": "Windsor-Essex County Health Unit",        "lat": 42.31, "lng": -83.03, "population": 422631,  "region": "West"},
    {"id": "MLHU", "name": "Middlesex-London Health Unit",            "lat": 43.00, "lng": -81.27, "population": 510800,  "region": "West"},
    {"id": "SMDHU","name": "Simcoe Muskoka District Health Unit",     "lat": 44.42, "lng": -79.66, "population": 595400,  "region": "Central"},
    {"id": "TBDHU","name": "Thunder Bay District Health Unit",        "lat": 48.38, "lng": -89.25, "population": 157700,  "region": "North"},
    {"id": "SDHU", "name": "Sudbury & District Health Unit",          "lat": 46.49, "lng": -81.01, "population": 200100,  "region": "North"},
    {"id": "KFL",  "name": "Kingston Public Health",                  "lat": 44.23, "lng": -76.49, "population": 195800,  "region": "East"},
    {"id": "PDH",  "name": "Peterborough Public Health",              "lat": 44.30, "lng": -78.32, "population": 155900,  "region": "East"},
    {"id": "WDGPH","name": "Wellington-Dufferin-Guelph Public Health","lat": 43.55, "lng": -80.26, "population": 337100,  "region": "West"},
    {"id": "NPH",  "name": "Niagara Region Public Health",            "lat": 43.10, "lng": -79.07, "population": 475200,  "region": "West"},
    {"id": "GBHU", "name": "Grey Bruce Health Unit",                  "lat": 44.57, "lng": -80.94, "population": 170100,  "region": "West"},
    {"id": "EOHU", "name": "Eastern Ontario Health Unit",             "lat": 45.00, "lng": -74.73, "population": 207000,  "region": "East"},
    {"id": "NBPSDHU","name": "North Bay Parry Sound District Health Unit","lat": 46.31,"lng":-79.46,"population": 130800,  "region": "North"},
    {"id": "APH",  "name": "Algoma Public Health",                    "lat": 46.51, "lng": -84.34, "population": 111400,  "region": "North"},
    {"id": "NWHU", "name": "Northwestern Health Unit",                "lat": 49.78, "lng": -92.84, "population": 76600,   "region": "North"},
    {"id": "HPEPH","name": "Hastings Prince Edward Public Health",    "lat": 44.56, "lng": -77.34, "population": 175000,  "region": "East"},
    {"id": "RCDHU","name": "Renfrew County and District Health Unit", "lat": 45.48, "lng": -76.68, "population": 107000,  "region": "East"},
]

DISEASES = [
    "Influenza A",
    "Influenza B",
    "COVID-19",
    "RSV",
    "Gastroenteritis",
    "Lyme Disease",
    "Salmonella",
    "Campylobacter",
    "West Nile Virus",
    "Mumps",
]

PHU_MAP = {p["id"]: p for p in PHUS}
