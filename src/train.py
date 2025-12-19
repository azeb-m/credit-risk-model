import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature  # ADDED

# Paths
DATA_PATH = os.path.join("data", "processed", "model_input.csv")

# Load processed features with target
print("📥 Loading processed data...")
df = pd.read_csv(DATA_PATH)

# Assuming 'is_high_risk' column exists
TARGET = "is_high_risk"
FEATURES = df.drop(columns=["CustomerId", TARGET]).columns

X = df[FEATURES]
y = df[TARGET]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data split: Train={X_train.shape}, Test={X_test.shape}")

# Start MLflow experiment
mlflow.set_experiment("Credit_Risk_Model")

print("\n" + "="*60)
print("🤖 BEFORE TUNING (Default Parameters)")
print("="*60)

# ============================================
# BEFORE TUNING - Logistic Regression
# ============================================
print("\n📈 Logistic Regression (No Tuning)")
with mlflow.start_run(run_name="Logistic_Regression_without_Tuning"):
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train, y_train)
    y_pred = model_lr.predict(X_test)
    y_proba = model_lr.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    # Log metrics and model
    mlflow.log_params(model_lr.get_params())
    mlflow.log_metrics(metrics)
    # UPDATED: Use standard artifact path
    signature = infer_signature(X_train, model_lr.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model_lr,
        artifact_path="model",  # Standard path instead of custom name
        signature=signature
    )

    print("LR Metrics:", metrics)
    lr_untuned_roc_auc = metrics["roc_auc"]

# ============================================
# BEFORE TUNING - Random Forest
# ============================================
print("\n🌲 Random Forest (No Tuning)")
with mlflow.start_run(run_name="Random_Forest_without_Tuning"):
    model_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    y_proba = model_rf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    mlflow.log_params(model_rf.get_params())
    mlflow.log_metrics(metrics)
    # UPDATED: Use standard artifact path
    signature = infer_signature(X_train, model_rf.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model_rf,
        artifact_path="model",  # Standard path instead of custom name
        signature=signature
    )

    print("RF Metrics:", metrics)
    rf_untuned_roc_auc = metrics["roc_auc"]

print("\n" + "="*60)
print("🎛️  HYPERPARAMETER TUNING (GridSearchCV)")
print("="*60)

# ============================================
# TUNING - Logistic Regression
# ============================================
print("\n🎯 Tuning Logistic Regression...")
with mlflow.start_run(run_name="Tuned_Logistic_Regression"):
    # Define parameter grid for Logistic Regression
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l2'],  # L2 regularization
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000, 2000],
        'class_weight': [None, 'balanced']
    }
    
    # Perform grid search
    grid_lr = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid_lr,
        cv=5,  # 5-fold cross-validation
        scoring='roc_auc',  # Optimize for ROC-AUC
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    print("Searching best parameters for Logistic Regression...")
    grid_lr.fit(X_train, y_train)
    
    # Get best model
    best_lr = grid_lr.best_estimator_
    y_pred = best_lr.predict(X_test)
    y_proba = best_lr.predict_proba(X_test)[:, 1]
    
    # Metrics for tuned model
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    # Log best parameters and metrics
    mlflow.log_params(grid_lr.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.log_metric("best_cv_score", grid_lr.best_score_)
    # UPDATED: Use standard artifact path and register directly
    signature = infer_signature(X_train, best_lr.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=best_lr,
        artifact_path="model",  # Standard path
        signature=signature,
        input_example=X_train.iloc[:5],
        registered_model_name="CreditRisk_Predictor"  # Register directly
    )
    
    print(f"✅ Best parameters: {grid_lr.best_params_}")
    print(f"✅ Best CV ROC-AUC: {grid_lr.best_score_:.4f}")
    print("LR Tuned Metrics:", metrics)
    lr_tuned_roc_auc = metrics["roc_auc"]

# ============================================
# TUNING - Random Forest
# ============================================
print("\n🎯 Tuning Random Forest...")
with mlflow.start_run(run_name="Tuned_Random_Forest"):
    # Define parameter grid for Random Forest
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Perform grid search
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Searching best parameters for Random Forest...")
    grid_rf.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_rf.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Metrics for tuned model
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    # Log best parameters and metrics
    mlflow.log_params(grid_rf.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.log_metric("best_cv_score", grid_rf.best_score_)
    # UPDATED: Use standard artifact path and register directly
    signature = infer_signature(X_train, best_rf.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=best_rf,
        artifact_path="model",  # Standard path
        signature=signature,
        input_example=X_train.iloc[:5],
        registered_model_name="CreditRisk_Predictor"  # Register directly
    )
    
    print(f"✅ Best parameters: {grid_rf.best_params_}")
    print(f"✅ Best CV ROC-AUC: {grid_rf.best_score_:.4f}")
    print("RF Tuned Metrics:", metrics)
    rf_tuned_roc_auc = metrics["roc_auc"]

print("\n" + "="*60)
print("📊 COMPARISON: BEFORE vs AFTER TUNING")
print("="*60)

# Create comparison DataFrame
comparison_data = {
    "Model": ["Logistic Regression", "Logistic Regression", 
              "Random Forest", "Random Forest"],
    "Tuning": ["No Tuning", "With Tuning", 
               "No Tuning", "With Tuning"],
    "ROC-AUC": [lr_untuned_roc_auc, lr_tuned_roc_auc,
                rf_untuned_roc_auc, rf_tuned_roc_auc],
    "Accuracy": [
        lr_untuned_roc_auc,  # Placeholder - you'd need to store accuracy too
        lr_tuned_roc_auc,    # Placeholder
        rf_untuned_roc_auc,  # Placeholder
        rf_tuned_roc_auc     # Placeholder
    ]
}

# Display comparison
print("\n📈 Performance Comparison (ROC-AUC):")
print("Model               | Tuning     | ROC-AUC   | Improvement")
print("-" * 50)
print(f"Logistic Regression | No Tuning  | {lr_untuned_roc_auc:.4f}   | -")
print(f"Logistic Regression | With Tuning| {lr_tuned_roc_auc:.4f}   | +{(lr_tuned_roc_auc - lr_untuned_roc_auc):.4f}")
print(f"Random Forest       | No Tuning  | {rf_untuned_roc_auc:.4f}   | -")
print(f"Random Forest       | With Tuning| {rf_tuned_roc_auc:.4f}   | +{(rf_tuned_roc_auc - rf_untuned_roc_auc):.4f}")

# Determine best model
best_overall_roc_auc = max(lr_tuned_roc_auc, rf_tuned_roc_auc)
if best_overall_roc_auc == lr_tuned_roc_auc:
    best_model = "Logistic Regression (Tuned)"
elif best_overall_roc_auc == rf_tuned_roc_auc:
    best_model = "Random Forest (Tuned)"

print("\n🏆 Best Overall Model:", best_model)
print(f"🏆 Best ROC-AUC: {best_overall_roc_auc:.4f}")

print("\n✅ All models trained, tuned, and logged to MLflow!")
print("🔍 Run 'mlflow ui' to view experiment results")

print("\n" + "="*60)
print("🏛️  REGISTERING BEST MODEL IN MLFLOW REGISTRY")
print("="*60)



# Initialize MLflow client
client = MlflowClient()

# Model registry name
model_name = "CreditRisk_Predictor"

# The best model run is "Random_Forest_Tuned" (Capitalized)
experiment = client.get_experiment_by_name("Credit_Risk_Model")

if experiment:
    print(f"✅ Found experiment: {experiment.name}")
    
    # Search for the exact run name - CASE SENSITIVE!
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'Tuned_Random_Forest'",  # Capitalized!
        max_results=1
    )
    
    if runs:
        best_run = runs[0]
        print(f"✅ Found best run: {best_run.info.run_id}")
        print(f"📊 Run Name: {best_run.data.tags.get('mlflow.runName')}")
        print(f"📊 Best ROC-AUC: {best_run.data.metrics.get('roc_auc', 0):.4f}")
        
        # UPDATED: Use standard artifact path
        artifact_path = "model"
        model_uri = f"runs:/{best_run.info.run_id}/{artifact_path}"
        
        print(f"🔍 Looking for model at: {model_uri}")
        
        try:
            # First, let's see what artifacts are available
            artifacts = client.list_artifacts(best_run.info.run_id)
            print("\n📁 Available artifacts in this run:")
            for artifact in artifacts:
                print(f"  - {artifact.path}")
            
            # Check if our expected artifact exists
            artifact_exists = any(artifact_path in artifact.path for artifact in artifacts)
            
            if artifact_exists:
                print(f"\n✅ Found artifact: {artifact_path}")
                
                # Create registered model if it doesn't exist
                try:
                    client.get_registered_model(model_name)
                    print(f"📚 Model '{model_name}' already exists in registry")
                except:
                    client.create_registered_model(model_name)
                    print(f"📚 Created new model registry: '{model_name}'")
                
                # Create model version
                model_version = client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    run_id=best_run.info.run_id
                )
                
                print(f"✅ Model version created: v{model_version.version}")
                
                # Transition to Production stage
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                
                print(f"✅ Model v{model_version.version} transitioned to 'Production'")
                print(f"🔗 Model URI: models:/{model_name}/Production")
                print(f"🔗 For inference: mlflow.pyfunc.load_model('models:/{model_name}/Production')")
                
            else:
                print(f"\n❌ Artifact '{artifact_path}' not found!")
                print("💡 Trying alternative artifact names...")
                
                # UPDATED: Simplified alternative paths
                alternative_paths = ["model"]  # Only try the standard path
                
                for alt_path in alternative_paths:
                    alt_uri = f"runs:/{best_run.info.run_id}/{alt_path}"
                    print(f"  Trying: {alt_path}")
                    
                    try:
                        # Try to create model version with this path
                        model_version = client.create_model_version(
                            name=model_name,
                            source=alt_uri,
                            run_id=best_run.info.run_id
                        )
                        
                        print(f"✅ Success with artifact: {alt_path}")
                        print(f"✅ Model version created: v{model_version.version}")
                        
                        # Transition to Production
                        client.transition_model_version_stage(
                            name=model_name,
                            version=model_version.version,
                            stage="Production"
                        )
                        
                        print(f"✅ Model v{model_version.version} transitioned to 'Production'")
                        break
                        
                    except Exception as e:
                        print(f"  ❌ Failed: {e}")
                        continue  # Try next alternative
                        
        except Exception as e:
            print(f"❌ Error: {e}")
            
    else:
        print("❌ Could not find 'Tuned_Random_Forest' run")
        print("\n🔍 Searching for available runs...")
        
        # List all runs to see what's available
        all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        print(f"Found {len(all_runs)} runs:")
        for run in all_runs:
            run_name = run.data.tags.get('mlflow.runName', 'Unnamed')
            roc_auc = run.data.metrics.get('roc_auc', 'N/A')
            print(f"  - '{run_name}' (ROC-AUC: {roc_auc})")
            
else:
    print("❌ Experiment 'Credit_Risk_Model' not found!")

print("\n" + "="*60)
print("✅ Registration process completed!")
print("="*60)