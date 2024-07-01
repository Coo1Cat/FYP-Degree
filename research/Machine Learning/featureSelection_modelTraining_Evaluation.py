import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_metrics_to_csv(metrics_dict, output_file):
    with open(output_file, 'w') as f:
        for model_name, metrics in metrics_dict.items():
            f.write(f"{model_name} Metrics\n")
            for metric, values in metrics.items():
                if isinstance(values, list) and len(values) > 1:
                    values_str = ','.join(map(str, values))
                    f.write(f"{metric},{values_str}\n")
                else:
                    f.write(f"{metric},{values[0]}\n")
            f.write("\n")

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, dataset_name, model_directory, metrics_dict, charts_directory):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
    print(f"{model_name} Cross-Validation Scores: {cv_scores}")
    print(f"{model_name} Mean Cross-Validation Score: {cv_scores.mean()}")

    model.fit(X_train, y_train)
    
    joblib_file = os.path.join(model_directory, f'{model_name}_{dataset_name}.pkl')
    joblib.dump(model, joblib_file)
    print(f"{model_name} model saved to {joblib_file}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{model_name} F1 Score: {f1:.2f}")

    print(f"\n{model_name} Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

    print(f"\n{model_name} True Positives (TP): {tp}")
    print(f"{model_name} False Positives (FP): {fp}")
    print(f"{model_name} True Negatives (TN): {tn}")
    print(f"{model_name} False Negatives (FN): {fn}")
    print(f"{model_name} Sensitivity (Recall): {sensitivity:.2f}")
    print(f"{model_name} False Positive Rate (FPR): {fpr:.2f}")
    print(f"{model_name} False Negative Rate (FNR): {fnr:.2f}")

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    fpr_values, tpr_values, thresholds = roc_curve(y_test, y_prob, pos_label=1)
    roc_auc = auc(fpr_values, tpr_values)
    print(f"\n{model_name} ROC Curve:")
    print("False Positive Rate:", fpr_values)
    print("True Positive Rate:", tpr_values)
    print("Thresholds:", thresholds)
    print(f"{model_name} AUC:", roc_auc)

    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob, pos_label=1)
    pr_auc = auc(recall, precision)
    print(f"\n{model_name} Precision-Recall Curve AUC: {pr_auc}")

    metrics_dict[model_name] = {
        'Cross-Validation Scores': list(cv_scores),
        'Mean Cross-Validation Score': [cv_scores.mean()],
        'Accuracy': [accuracy],
        'F1 Score': [f1],
        'Classification Report': [report],
        'Confusion Matrix': [cm.tolist()],
        'True Positives': [tp],
        'False Positives': [fp],
        'True Negatives': [tn],
        'False Negatives': [fn],
        'Sensitivity': [sensitivity],
        'False Positive Rate': [fpr],
        'False Negative Rate': [fnr],
        'AUC': [roc_auc],
        'Precision-Recall AUC': [pr_auc]
    }

    # Plot ROC curve
    plot_roc_curve(fpr_values, tpr_values, roc_auc, model_name, dataset_name, charts_directory)
    
    # Plot Precision-Recall curve
    plot_precision_recall_curve(recall, precision, pr_auc, model_name, dataset_name, charts_directory)

    # Plot confusion matrix heatmap
    plot_confusion_matrix(cm, model_name, dataset_name, charts_directory)

def process_data(dataset, test_size):
    try:
        data = pd.read_csv(dataset)
    except FileNotFoundError:
        print(f"Dataset not found at path: {dataset}")
        return None, None, None, None
    except pd.errors.EmptyDataError:
        print(f"Dataset is empty at path: {dataset}")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None, None, None, None

    phishing_count = data[data['label'] == 1].shape[0]
    legit_count = data[data['label'] == 0].shape[0]
    print(f"Number of phishing URLs: {phishing_count}")
    print(f"Number of legitimate URLs: {legit_count}")

    if phishing_count > 2 * legit_count or legit_count > 2 * phishing_count:
        print("The dataset is imbalanced. Consider using techniques to balance the dataset.")

    try:
        columns_to_drop = ['label', 'URL']
        columns_to_drop = [col for col in columns_to_drop if col in data.columns]
        X = data.drop(columns=columns_to_drop)
        y = data['label']
    except KeyError as e:
        print(f"Column missing in the dataset: {e}")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=43)
    smote = SMOTE(random_state=43)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test, data, X

def random_forest(X_train, X_test, y_train, y_test, dataset_name, model_directory, metrics_dict, charts_directory):
    rf_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=43)
    evaluate_model(rf_classifier, X_train, y_train, X_test, y_test, "RandomForest", dataset_name, model_directory, metrics_dict, charts_directory)
    return rf_classifier

def linear_svm(X_train, X_test, y_train, y_test, dataset_name, model_directory, metrics_dict, charts_directory):
    linear_svm_classifier = LinearSVC(C=1.0, max_iter=10000, random_state=42)
    evaluate_model(linear_svm_classifier, X_train, y_train, X_test, y_test, "Linear_SVM", dataset_name, model_directory, metrics_dict, charts_directory)

def knn(X_train, X_test, y_train, y_test, dataset_name, model_directory, metrics_dict, charts_directory):
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    evaluate_model(knn_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "KNN", dataset_name, model_directory, metrics_dict, charts_directory)

def feature_importance(rf_classifier, X, data, dataset_name, charts_directory, importance_threshold=0.01):
    importances = rf_classifier.feature_importances_
    feature_names = X.columns
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    feature_importances_file = os.path.join(charts_directory, f'{dataset_name}_feature_importances.txt')
    os.makedirs(os.path.dirname(feature_importances_file), exist_ok=True)
    with open(feature_importances_file, 'w') as f:
        for idx, row in feature_importances.iterrows():
            f.write(f"Feature: {row['Feature']}, Importance: {row['Importance']:.6f}\n")
    print(f"Feature importances saved to {feature_importances_file}")

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
    plt.title('Feature Importances from RandomForestClassifier')
    plt.savefig(os.path.join(charts_directory, f'{dataset_name}_Feature_Importances.png'))
    plt.close()

    important_features = feature_importances[feature_importances['Importance'] > importance_threshold]['Feature'].tolist()
    important_features_with_label = important_features + ['label']

    data_important_features = data[important_features_with_label]

    important_features_dataset_path = os.path.join(os.path.dirname(__file__), 'cleaned_dataset', f'refined_{dataset_name}.csv')
    data_important_features.to_csv(important_features_dataset_path, index=False)
    print(f"Dataset with important features saved to '{important_features_dataset_path}'")

def plot_roc_curve(fpr_values, tpr_values, roc_auc, model_name, dataset_name, charts_directory):
    plt.figure()
    plt.plot(fpr_values, tpr_values, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(charts_directory, f'{model_name}_{dataset_name}_ROC_Curve.png'))
    plt.close()

def plot_precision_recall_curve(recall, precision, pr_auc, model_name, dataset_name, charts_directory):
    plt.figure()
    plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(charts_directory, f'{model_name}_{dataset_name}_Precision_Recall_Curve.png'))
    plt.close()

def plot_confusion_matrix(cm, model_name, dataset_name, charts_directory):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(charts_directory, f'{model_name}_{dataset_name}_Confusion_Matrix.png'))
    plt.close()

def plot_comparison_charts(metrics_dict, charts_directory):
    # Accuracy comparison
    accuracies = {model: metrics['Accuracy'][0] for model, metrics in metrics_dict.items()}
    models = list(accuracies.keys())
    accuracy_values = list(accuracies.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=accuracy_values, palette='Set2')
    plt.title('Model Comparison - Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.0)  
    plt.xlabel('Models')
    plt.savefig(os.path.join(charts_directory, 'Model_Comparison_Accuracy.png'))
    plt.close()

    # Sensitivity comparison
    sensitivities = {model: metrics['Sensitivity'][0] for model, metrics in metrics_dict.items()}
    sensitivity_values = list(sensitivities.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=sensitivity_values, palette='Set1')
    plt.title('Model Comparison - Sensitivity (Recall)')
    plt.ylabel('Sensitivity (Recall)')
    plt.ylim(0.0, 1.0)  
    plt.xlabel('Models')
    plt.savefig(os.path.join(charts_directory, 'Model_Comparison_Sensitivity.png'))
    plt.close()

    # False Positive Rate comparison
    false_positive_rates = {model: metrics['False Positive Rate'][0] for model, metrics in metrics_dict.items()}
    false_positive_rate_values = list(false_positive_rates.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=false_positive_rate_values, palette='Pastel1')
    plt.title('Model Comparison - False Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.ylim(0.0, 1.0)  
    plt.xlabel('Models')
    plt.savefig(os.path.join(charts_directory, 'Model_Comparison_False_Positive_Rate.png'))
    plt.close()

    # False Negative Rate comparison
    false_negative_rates = {model: metrics['False Negative Rate'][0] for model, metrics in metrics_dict.items()}
    false_negative_rate_values = list(false_negative_rates.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=false_negative_rate_values, palette='Pastel2')
    plt.title('Model Comparison - False Negative Rate')
    plt.ylabel('False Negative Rate')
    plt.ylim(0.0, 1.0)  
    plt.xlabel('Models')
    plt.savefig(os.path.join(charts_directory, 'Model_Comparison_False_Negative_Rate.png'))
    plt.close()

    # AUC comparison
    aucs = {model: metrics['AUC'][0] for model, metrics in metrics_dict.items()}
    auc_values = list(aucs.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=auc_values, palette='Set3')
    plt.title('Model Comparison - AUC')
    plt.ylabel('AUC')
    plt.ylim(0.0, 1.0)  
    plt.xlabel('Models')
    plt.savefig(os.path.join(charts_directory, 'Model_Comparison_AUC.png'))
    plt.close()

    # Precision-Recall AUC comparison
    pr_aucs = {model: metrics['Precision-Recall AUC'][0] for model, metrics in metrics_dict.items()}
    pr_auc_values = list(pr_aucs.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=pr_auc_values, palette='coolwarm')
    plt.title('Model Comparison - Precision-Recall AUC')
    plt.ylabel('Precision-Recall AUC')
    plt.ylim(0.0, 1.0)  
    plt.xlabel('Models')
    plt.savefig(os.path.join(charts_directory, 'Model_Comparison_Precision_Recall_AUC.png'))
    plt.close()

def main():
    dataset = os.path.join(os.path.dirname(__file__), 'TBD', 'onlyContent_featureExtract_selfCollect_withMinus.csv')
    test_size = 0.2

    X_train, X_test, y_train, y_test, data, X = process_data(dataset, test_size)
    if X_train is None:
        return
    
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]
    model_directory = os.path.join(os.path.dirname(__file__), 'trained_model')
    charts_directory = os.path.join(model_directory, 'charts')
    os.makedirs(model_directory, exist_ok=True)
    os.makedirs(charts_directory, exist_ok=True)

    metrics_dict = {}

    rf_classifier = random_forest(X_train, X_test, y_train, y_test, dataset_name, model_directory, metrics_dict, charts_directory)
    knn(X_train, X_test, y_train, y_test, dataset_name, model_directory, metrics_dict, charts_directory)
    linear_svm(X_train, X_test, y_train, y_test, dataset_name, model_directory, metrics_dict, charts_directory)

    feature_importance(rf_classifier, X, data, dataset_name, charts_directory)

    metrics_output_file = os.path.join(model_directory, f'{dataset_name}_metrics.csv')
    save_metrics_to_csv(metrics_dict, metrics_output_file)
    print(f"All metrics saved to {metrics_output_file}")

    plot_comparison_charts(metrics_dict, charts_directory)

if __name__ == '__main__':
    main()
