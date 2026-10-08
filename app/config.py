#Step 1: Project Configuration

from pathlib import Path


#Step 2: Defining Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

EVALUATION_DIR = MODELS_DIR / "evaluation"

OPTIMIZATION_DIR = MODELS_DIR / "optimization"

EXPLAINABILITY_DIR = MODELS_DIR / "explainability"

BUSINESS_INSIGHTS_DIR = MODELS_DIR / "business_insights"


#Step 3: Defining Model Files

OPTIMIZED_MODEL_PATH = (
    OPTIMIZATION_DIR /
    "optimized_airline_satisfaction_ann.keras"
)

PREPROCESSOR_PATH = (
    PROCESSED_DIR /
    "preprocessor.joblib"
)

FEATURE_NAMES_PATH = (
    PROCESSED_DIR /
    "feature_names.npy"
)


#Step 4: Defining Analysis Files

EVALUATION_PATH = (
    EVALUATION_DIR /
    "evaluation_summary.csv"
)

THRESHOLD_ANALYSIS_PATH = (
    EVALUATION_DIR /
    "threshold_analysis.csv"
)

PREDICTIONS_PATH = (
    EVALUATION_DIR /
    "test_predictions.csv"
)

OPTIMIZATION_RESULTS_PATH = (
    OPTIMIZATION_DIR /
    "optimization_experiments.csv"
)

BASELINE_VS_OPTIMIZED_PATH = (
    OPTIMIZATION_DIR /
    "baseline_vs_optimized.csv"
)

PERMUTATION_IMPORTANCE_PATH = (
    EXPLAINABILITY_DIR /
    "permutation_importance.csv"
)

TOP_FEATURES_PATH = (
    EXPLAINABILITY_DIR /
    "top_features.csv"
)

PASSENGER_EXPLANATIONS_PATH = (
    EXPLAINABILITY_DIR /
    "passenger_explanations.csv"
)

BUSINESS_RECOMMENDATIONS_PATH = (
    BUSINESS_INSIGHTS_DIR /
    "business_recommendations.csv"
)


#Step 5: Model Configuration

PREDICTION_THRESHOLD = 0.50

PROJECT_NAME = "AirlineIQ"

MODEL_NAME = "Optimized Artificial Neural Network"

TARGET_NAME = "Passenger Satisfaction"