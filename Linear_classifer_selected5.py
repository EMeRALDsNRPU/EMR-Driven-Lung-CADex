import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Create a new results directory
results_dir = 'results10'
os.makedirs(results_dir, exist_ok=True)

# Load datasets
try:
    # Load all radiomic features (8 features + label = 9 columns)
    data_features = pd.read_csv('cleaned_data_for_linear_classifer.csv', header=None, skiprows=1)
    print(f"Loaded features data with shape: {data_features.shape}")
    
    # Load EMR data
    data_emr = pd.read_csv('malignancy_emr.csv')
    print(f"Loaded EMR data with shape: {data_emr.shape}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# All radiomic features in the file
all_radiomic_features = ['subtlety', 'internal_structure', 'calcification', 'roundness', 
                         'margin', 'lobulation', 'spiculation', 'internal_texture']

# Assign column names to features data (8 features + label)
data_features.columns = all_radiomic_features + ['label']
print("All columns in features data:", data_features.columns.tolist())

# Selected radiomic features (subset we want to work with)
selected_radiomic_features = ['subtlety', 'calcification', 'margin', 'lobulation', 'internal_texture']
emr_features = ['Age', 'Gender', 'Smoking_Status', 'Alcohol_Status']

# Convert feature columns to float
data_features[all_radiomic_features] = data_features[all_radiomic_features].astype(float)
data_emr[all_radiomic_features] = data_emr[all_radiomic_features].astype(float)

# Merge datasets on all radiomic features first
combined_data = pd.merge(data_features, data_emr, on=all_radiomic_features, how='inner')

# Now select only the features we want to use
required_columns = selected_radiomic_features + emr_features + ['label']
combined_data = combined_data[required_columns]
print(f"Final dataset shape with selected features: {combined_data.shape}")

# Add feature engineering specific to selected features
combined_data['margin_texture'] = combined_data['margin'] * combined_data['internal_texture']
combined_data['calcification_texture'] = combined_data['calcification'] * combined_data['internal_texture']
combined_data['subtlety_margin'] = combined_data['subtlety'] * combined_data['margin']
combined_data['texture_calcification_ratio'] = combined_data['internal_texture'] / (combined_data['calcification'] + 0.1)

# Create polynomial features for key indicators
for col in ['subtlety', 'margin']:
    combined_data[f'{col}_squared'] = combined_data[col] ** 2

# Add risk scores based on domain knowledge
combined_data['malignancy_risk_score'] = (
    combined_data['margin'] * 1.5 + 
    combined_data['subtlety'] * 1.2 + 
    combined_data['calcification'] * 1.0 +
    combined_data['internal_texture'] * 0.8
)

# Analyze feature importance
X_initial = combined_data.drop(columns=['label'])
y = combined_data['label'].values

# Create correlation matrix
correlation = X_initial.corrwith(pd.Series(y))
correlation_df = pd.DataFrame({'Feature': correlation.index, 'Correlation': correlation.values})
correlation_df = correlation_df.sort_values('Correlation', ascending=False)

# Save correlation analysis
correlation_df.to_csv(os.path.join(results_dir, 'feature_correlation.csv'), index=False)

# Select top features based on correlation
top_features = correlation_df['Feature'].iloc[:15].tolist()  # Select top 15 features

# Extract selected features
X = combined_data[top_features].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features with robust scaler
scaler = PowerTransformer(method='yeo-johnson')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Increase weight for minority class even more
class_weight_dict[1] *= 1.5

# Handle class imbalance with targeted SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Use even higher threshold for classification
classification_threshold = 0.75

# Define all requested models with different configurations
models = {
    # Gradient Boosting
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.7,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42
    ),
    
    # SVM Models
    'svm_linear': SVC(
        kernel='linear',
        probability=True,
        class_weight=class_weight_dict,
        C=0.1,
        cache_size=1000,
        random_state=42
    ),
    'svm_quadratic': make_pipeline(
        PolynomialFeatures(degree=2),
        SVC(
            kernel='linear',
            probability=True,
            class_weight=class_weight_dict,
            C=0.1,
            cache_size=1000,
            random_state=42
        )
    ),
    'svm_rbf': SVC(
        kernel='rbf',
        probability=True,
        class_weight=class_weight_dict,
        C=0.3,
        gamma='auto',
        cache_size=1000,
        random_state=42
    ),
    'cubic_svm': SVC(
        kernel='poly',
        degree=3,
        probability=True,
        class_weight=class_weight_dict,
        C=0.1,
        cache_size=1000,
        random_state=42
    ),
    'fine_svm': SVC(
        kernel='rbf',
        probability=True,
        class_weight=class_weight_dict,
        C=1.0,
        gamma='scale',
        cache_size=1000,
        random_state=42
    ),
    'medium_svm': SVC(
        kernel='rbf',
        probability=True,
        class_weight=class_weight_dict,
        C=0.5,
        gamma='scale',
        cache_size=1000,
        random_state=42
    ),
    
    # Random Forest
    'random_forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=8,
        min_samples_split=15,
        max_features='sqrt',
        class_weight=class_weight_dict,
        bootstrap=True,
        random_state=42
    ),
    
    # KNN Models
    'knn': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        p=2  # Euclidean distance
    ),
    'cubic_knn': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        p=3  # Minkowski with p=3
    ),
    'fine_knn': KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        p=2
    ),
    'medium_knn': KNeighborsClassifier(
        n_neighbors=10,
        weights='distance',
        p=2
    ),
    
    # Decision Trees
    'decision_tree': DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weight_dict,
        random_state=42
    ),
    'fine_tree': DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight_dict,
        random_state=42
    ),
    'medium_tree': DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight=class_weight_dict,
        random_state=42
    ),
    
    # Logistic Regression
    'logistic_regression': LogisticRegression(
        solver='saga',
        class_weight=class_weight_dict,
        C=0.05,
        penalty='elasticnet',
        l1_ratio=0.8,
        max_iter=5000,
        random_state=42
    )
}

# Create a voting classifier with the best performing models
hybrid_models = {
    **models,
    'voting_classifier': VotingClassifier(estimators=[
        ('gb', models['gradient_boosting']),
        ('rf', models['random_forest']),
        ('svm', models['svm_rbf'])
    ], voting='soft')
}

results_table = []
model_collection = []

# Train and evaluate all models
for model_name, model in hybrid_models.items():
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Training {model_name}...")
    
    try:
        # Handle pipeline models differently
        if 'pipeline' in str(type(model)).lower():
            model.fit(X_train, y_train_resampled)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba >= classification_threshold).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = None
        else:
            model.fit(X_train_resampled, y_train_resampled)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                y_pred = (y_pred_proba >= classification_threshold).astype(int)
            else:
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC AUC if possible
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            # Save ROC curve plot
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic - {model_name}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(model_dir, 'roc_curve.png'))
            plt.close()
        else:
            roc_auc = None
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        np.savetxt(os.path.join(model_dir, 'confusion_matrix_values.txt'), conf_matrix, fmt='%d', delimiter=',')
        np.savetxt(os.path.join(model_dir, 'confusion_matrix_percent.txt'), conf_matrix_percent, fmt='%.2f', delimiter=',')
        
        # Save confusion matrix images
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (Values) - {model_name}')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')
        plt.savefig(os.path.join(model_dir, 'confusion_matrix_values.png'))
        plt.close()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix_percent, interpolation='nearest', cmap=plt.cm.Greens, vmin=0, vmax=100)
        plt.title(f'Confusion Matrix (Percent) - {model_name}')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{conf_matrix_percent[i, j]:.2f}%', ha='center', va='center', color='black')
        plt.savefig(os.path.join(model_dir, 'confusion_matrix_percent.png'))
        plt.close()
        
        # Save precision-recall curve
        if y_pred_proba is not None:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
            average_precision = average_precision_score(y_test, y_pred_proba)
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall_curve, precision_curve, lw=2, label=f'AP = {average_precision:.3f}')
            plt.axhline(y=precision, color='r', linestyle='--', label=f'Current Precision = {precision:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(model_dir, 'precision_recall_curve.png'))
            plt.close()
            
            # Save threshold analysis
            thresholds_results = []
            for thresh in np.arange(0.3, 0.95, 0.05):
                y_pred_t = (y_pred_proba >= thresh).astype(int)
                prec = precision_score(y_test, y_pred_t)
                rec = recall_score(y_test, y_pred_t)
                thresholds_results.append([thresh, prec, rec])
                
            thresholds_df = pd.DataFrame(thresholds_results, columns=['Threshold', 'Precision', 'Recall'])
            thresholds_df.to_csv(os.path.join(model_dir, 'threshold_analysis.csv'), index=False)
        
        # Save model and scaler
        joblib.dump(model, os.path.join(model_dir, f'{model_name}_model.joblib'))
        joblib.dump(scaler, os.path.join(model_dir, f'{model_name}_scaler.joblib'))
        
        # Save feature names
        with open(os.path.join(model_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(top_features))
        
        # Save all metrics
        metrics_filename = os.path.join(model_dir, 'evaluation_metrics.txt')
        with open(metrics_filename, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall (Sensitivity): {recall:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            if roc_auc is not None:
                f.write(f"ROC AUC: {roc_auc:.4f}\n")
            f.write(f"Positive Predictive Value: {ppv:.4f}\n")
            f.write(f"Negative Predictive Value: {npv:.4f}\n")
            if hasattr(model, 'predict_proba'):
                f.write(f"Classification Threshold: {classification_threshold:.2f}\n")
        
        # Store for results table
        model_results = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc if roc_auc is not None else 0,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv,
            'True Positives': tp,
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn
        }
        
        results_table.append(model_results)
        model_collection.append((model_name, model_results))
        
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        continue

# Convert results to DataFrame and save
results_df = pd.DataFrame(results_table)
results_df.to_csv(os.path.join(results_dir, 'model_results.csv'), index=False)

# Calculate collective best model using a weighted score
def calculate_collective_score(model_metrics):
    return (
        0.4 * model_metrics['Precision'] +
        0.2 * model_metrics['Recall'] +
        0.15 * model_metrics['Specificity'] +
        0.15 * model_metrics['F1 Score'] +
        0.1 * model_metrics['Accuracy']
    )

# Add collective score to results
for i, model_data in enumerate(model_collection):
    model_name, metrics = model_data
    collective_score = calculate_collective_score(metrics)
    results_df.loc[i, 'Collective Score'] = collective_score

# Sort by collective score
results_df = results_df.sort_values('Collective Score', ascending=False)
results_df.to_csv(os.path.join(results_dir, 'model_results_ranked.csv'), index=False)

# Find best model based on collective score
best_model_idx = results_df['Collective Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_collective_score = results_df.loc[best_model_idx, 'Collective Score']
best_precision = results_df.loc[best_model_idx, 'Precision']

# Also identify model with highest precision
highest_precision_idx = results_df['Precision'].idxmax()
highest_precision_model = results_df.loc[highest_precision_idx, 'Model']
highest_precision_value = results_df.loc[highest_precision_idx, 'Precision']

# Save best model summary
with open(os.path.join(results_dir, 'best_model_summary.txt'), 'w') as f:
    f.write(f"Best overall model: {best_model_name}\n")
    f.write(f"Collective score: {best_collective_score:.4f}\n")
    f.write(f"Precision: {best_precision:.4f}\n\n")
    
    f.write(f"Highest precision model: {highest_precision_model}\n")
    f.write(f"Precision: {highest_precision_value:.4f}\n\n")
    
    f.write(f"All models ranked by collective score:\n")
    for idx, row in results_df.iterrows():
        f.write(f"\n{row['Model']}:\n")
        f.write(f"  Collective Score: {row['Collective Score']:.4f}\n")
        f.write(f"  Precision: {row['Precision']:.4f}\n")
        f.write(f"  Recall: {row['Recall']:.4f}\n")
        f.write(f"  F1 Score: {row['F1 Score']:.4f}\n")
        f.write(f"  Specificity: {row['Specificity']:.4f}\n")

# Create visualization of model comparison
plt.figure(figsize=(12, 10))
models_to_plot = results_df['Model'][:10]  # Top 10 models
metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Specificity', 'Accuracy']

for i, metric in enumerate(metrics_to_plot):
    plt.subplot(len(metrics_to_plot), 1, i+1)
    values = results_df.loc[results_df['Model'].isin(models_to_plot), metric].values
    plt.barh(models_to_plot, values)
    plt.title(f'Model Comparison - {metric}')
    plt.xlim(0, 1)
    for index, value in enumerate(values):
        plt.text(value + 0.01, index, f'{value:.3f}')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
plt.close()

print(f"\nAll models trained and results saved in the '{results_dir}' folder.")
print(f"Collective best model: {best_model_name} with score: {best_collective_score:.4f}")
print(f"Highest precision model: {highest_precision_model} with precision: {highest_precision_value:.4f}")