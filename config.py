from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
TELEMETRY_DIR = OUTPUT_DIR / "telemetry"
RUNS_DIR = OUTPUT_DIR / "runs"
VERIFY_DIR = OUTPUT_DIR / "verify"

for path in (CACHE_DIR, OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, TELEMETRY_DIR, RUNS_DIR, VERIFY_DIR):
    path.mkdir(parents=True, exist_ok=True)

# Time-based split
TRAIN_YEARS = list(range(2018, 2023))
VAL_YEAR = 2023
TEST_YEAR = 2024
ALL_SEASONS = TRAIN_YEARS + [VAL_YEAR, TEST_YEAR]

# Prediction task
TARGET_COL = "finish_position"
DNF_SENTINEL = 21
RANDOM_SEED = 42

# Feature engineering controls
N_CLEAN_LAPS = 7
MIN_STINT_LAPS = 4
RECENT_FORM_WINDOW = 5
TEAM_FORM_WINDOW = 8
TRACK_HISTORY_WINDOW = 3
DNF_HISTORY_WINDOW = 10
ENABLE_TELEMETRY_EXPLAINER = True
TWO_STAGE_DNF_THRESHOLD = 0.55
DEFAULT_DASHBOARD_MODEL = "two_stage"

# Model tuning
XGB_PARAM_SPACE = {
    "n_estimators": (150, 700),
    "max_depth": (3, 8),
    "learning_rate": (0.02, 0.2),
    "subsample": (0.65, 1.0),
    "colsample_bytree": (0.65, 1.0),
    "min_child_weight": (1, 8),
}
OPTUNA_N_TRIALS = 40

# Domain maps
STREET_CIRCUITS = {
    "Melbourne",
    "Baku",
    "Monaco",
    "Singapore",
    "Jeddah",
    "Miami",
    "Las Vegas",
}

POWER_CIRCUITS = {
    "Monza",
    "Spielberg",
    "Spa",
    "Montreal",
    "Jeddah",
    "Las Vegas",
}

HIGH_DOWNFORCE_CIRCUITS = {
    "Monaco",
    "Budapest",
    "Zandvoort",
    "Singapore",
    "Barcelona",
}

OVERTAKE_DIFFICULTY = {
    "Melbourne": 0.50,
    "Sakhir": 0.30,
    "Shanghai": 0.45,
    "Baku": 0.55,
    "Barcelona": 0.65,
    "Monaco": 0.95,
    "Montreal": 0.40,
    "Paul Ricard": 0.35,
    "Spielberg": 0.30,
    "Silverstone": 0.35,
    "Hockenheim": 0.40,
    "Budapest": 0.85,
    "Spa": 0.30,
    "Monza": 0.20,
    "Singapore": 0.85,
    "Sochi": 0.45,
    "Suzuka": 0.70,
    "Austin": 0.40,
    "Mexico City": 0.35,
    "Interlagos": 0.45,
    "Yas Island": 0.55,
    "Mugello": 0.40,
    "Imola": 0.72,
    "Portimao": 0.45,
    "Istanbul": 0.42,
    "Doha": 0.55,
    "Jeddah": 0.50,
    "Miami": 0.45,
    "Zandvoort": 0.78,
    "Lusail": 0.55,
    "Las Vegas": 0.25,
}

# Feature schema
NUMERIC_FEATURES = [
    "grid_position",
    "quali_gap_to_pole",
    "lap_consistency_std",
    "tyre_deg_slope",
    "stint_count",
    "air_temp",
    "track_temp",
    "temp_delta",
    "humidity",
    "driver_recent_avg_pos",
    "constructor_dnf_rate",
    "driver_recent_form_score",
    "constructor_recent_form_score",
    "track_specific_driver_score",
    "wet_performance_score",
    "street_circuit_score",
    "overtake_difficulty_factor",
    "pit_stop_efficiency",
    "safety_car_sensitivity",
    "quali_to_race_conversion",
    "dnf_risk_score",
    "avg_top_speed_last_5",
    "straight_line_speed_percentile_last_5",
    "avg_deg_same_compound_last_5",
    "constructor_strategy_score",
    "field_strength",
    "chaos_index",
    "avg_speed_trap",
    "speed_trap_percentile",
    "pit_stop_count",
    "strategy_delta",
]

CATEGORICAL_FEATURES = [
    "compound_start",
    "circuit",
    "circuit_type",
    "season_phase",
]

BINARY_FEATURES = [
    "is_wet",
    "is_street_circuit",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

FEATURE_GROUPS = {
    "baseline_core": [
        "grid_position",
        "quali_gap_to_pole",
        "lap_consistency_std",
        "tyre_deg_slope",
        "stint_count",
        "compound_start",
        "circuit",
    ],
    "recent_form_stack": [
        "driver_recent_avg_pos",
        "constructor_dnf_rate",
        "driver_recent_form_score",
        "constructor_recent_form_score",
        "quali_to_race_conversion",
        "dnf_risk_score",
    ],
    "track_weather_stack": [
        "air_temp",
        "track_temp",
        "temp_delta",
        "humidity",
        "is_wet",
        "circuit_type",
        "season_phase",
        "track_specific_driver_score",
        "wet_performance_score",
        "street_circuit_score",
        "overtake_difficulty_factor",
        "is_street_circuit",
        "field_strength",
        "chaos_index",
    ],
    "telemetry_history_stack": [
        "pit_stop_efficiency",
        "safety_car_sensitivity",
        "avg_top_speed_last_5",
        "straight_line_speed_percentile_last_5",
        "avg_deg_same_compound_last_5",
        "constructor_strategy_score",
        "avg_speed_trap",
        "speed_trap_percentile",
        "pit_stop_count",
        "strategy_delta",
    ],
}
