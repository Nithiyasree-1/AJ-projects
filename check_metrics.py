import pickle
with open('anemia_prediction_model.pkl', 'rb') as f:
    data = pickle.load(f)
print("--- FINAL REPORT ---")
print(f"Best Model Name: {data['best_model_name']}")
print(f"Final Model Accuracy: {data['metrics']['Accuracy']:.4f}")
print(f"ROC-AUC Score: {data['metrics'].get('AUC', 0):.4f}")
