from ultralytics import YOLO
from multiprocessing import freeze_support
import os

best_model_path = r'path_to_model'
Yaml = os.path.join(r'path_to_dataset', 'training_index.yaml')

def run_quantitative_evaluation(model_path, yaml_path):
    model = YOLO(model_path)
    
    metrics = model.val(
        data = yaml_path,
        split = 'test',
        imgsz = 256,
        batch = 8,
        device = 'cuda',
        workers = 0,
        name = whichone + '_test'
    )
    
    print(f"Test Set mAP50:    {metrics.box.map50:.4f}")
    print(f"Test Set mAP50-95: {metrics.box.map:.4f}")
    return metrics

if __name__ == "__main__":
    freeze_support() 
    run_quantitative_evaluation(best_model_path, Yaml)
