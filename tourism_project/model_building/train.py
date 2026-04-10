import os
import pandas as pd
import joblib
import mlflow
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo, hf_hub_download

dataset_repo = "novicetopper/wellness-tourism-data"
model_repo = "novicetopper/wellness-tourism-model"
api = HfApi(token=os.getenv("HF_TOKEN"))

X_train_file = hf_hub_download(repo_id=dataset_repo, repo_type="dataset", filename="X_train.csv", token=os.getenv("HF_TOKEN"))
X_test_file = hf_hub_download(repo_id=dataset_repo, repo_type="dataset", filename="X_test.csv", token=os.getenv("HF_TOKEN"))
y_train_file = hf_hub_download(repo_id=dataset_repo, repo_type="dataset", filename="y_train.csv", token=os.getenv("HF_TOKEN"))
y_test_file = hf_hub_download(repo_id=dataset_repo, repo_type="dataset", filename="y_test.csv", token=os.getenv("HF_TOKEN"))

X_train = pd.read_csv(X_train_file)
X_test = pd.read_csv(X_test_file)
y_train = pd.read_csv(y_train_file).squeeze("columns")
y_test = pd.read_csv(y_test_file).squeeze("columns")

numeric_features = [
    "Age",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

categorical_features = [
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
pipeline = make_pipeline(preprocessor, model)

param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [3, 4, 5],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.4, 0.6, 0.8],
    "xgbclassifier__subsample": [0.7, 0.8, 1.0],
}

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("wellness-tourism-experiment")

with mlflow.start_run():
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        params = results["params"][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metrics({
        "test_accuracy": report["accuracy"],
        "test_precision": report["1"]["precision"],
        "test_recall": report["1"]["recall"],
        "test_f1_score": report["1"]["f1-score"],
    })

    model_path = "tourism_project/model_building/wellness_tourism_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print("Saved best model locally at", model_path)

    try:
        api.repo_info(repo_id=model_repo, repo_type="model")
        print(f"Model repository '{model_repo}' already exists.")
    except Exception:
        create_repo(repo_id=model_repo, repo_type="model", private=False)
        print(f"Created model repository '{model_repo}'.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="wellness_tourism_model.joblib",
        repo_id=model_repo,
        repo_type="model",
    )
    print("Uploaded the best model to the Hugging Face model hub.")
