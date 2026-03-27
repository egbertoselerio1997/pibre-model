"""Machine learning model modules live in this package."""

from .adaboost_regressor import (
	load_adaboost_regressor_params,
	predict_adaboost_regressor_model,
	run_adaboost_regressor_pipeline,
)
from .catboost_regressor import (
	load_catboost_regressor_params,
	predict_catboost_regressor_model,
	run_catboost_regressor_pipeline,
)
from .cobre import load_cobre_params, predict_cobre_model, run_cobre_pipeline, train_cobre_model
from .lightgbm_regressor import (
	load_lightgbm_regressor_params,
	predict_lightgbm_regressor_model,
	run_lightgbm_regressor_pipeline,
)
from .uncobre import (
	load_uncobre_params,
	predict_uncobre_model,
	run_uncobre_pipeline,
	train_uncobre_model,
)
from .random_forest_regressor import (
	load_random_forest_regressor_params,
	predict_random_forest_regressor_model,
	run_random_forest_regressor_pipeline,
)
from .svr_regressor import load_svr_regressor_params, predict_svr_regressor_model, run_svr_regressor_pipeline
from .xgboost_regressor import (
	load_xgboost_regressor_params,
	predict_xgboost_regressor_model,
	run_xgboost_regressor_pipeline,
)

__all__ = [
	"load_adaboost_regressor_params",
	"load_catboost_regressor_params",
	"load_cobre_params",
	"load_lightgbm_regressor_params",
	"load_uncobre_params",
	"load_random_forest_regressor_params",
	"load_svr_regressor_params",
	"load_xgboost_regressor_params",
	"predict_adaboost_regressor_model",
	"predict_catboost_regressor_model",
	"predict_cobre_model",
	"predict_lightgbm_regressor_model",
	"predict_uncobre_model",
	"predict_random_forest_regressor_model",
	"predict_svr_regressor_model",
	"predict_xgboost_regressor_model",
	"run_adaboost_regressor_pipeline",
	"run_catboost_regressor_pipeline",
	"run_cobre_pipeline",
	"run_lightgbm_regressor_pipeline",
	"run_uncobre_pipeline",
	"run_random_forest_regressor_pipeline",
	"run_svr_regressor_pipeline",
	"run_xgboost_regressor_pipeline",
	"train_cobre_model",
	"train_uncobre_model",
]
