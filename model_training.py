from ultralytics import YOLO
from multiprocessing import freeze_support
import os
import random

def main():
    model = YOLO("yolov8m.pt")
    
    Yaml = os.path.join(r'path_to_dataset', 'training_index.yaml')

    # Generate a random integer to use as the seed
    # current_seed = random.randint(0, 100000)
    # print(f"Starting training with random seed: {current_seed}")

    results = model.train(
        data = Yaml,
        epochs = 30,
        imgsz = 256,
        batch = 8,
        device = 'cuda',
        optimizer = "Adam",   
        name = "8m_30epoch",
        lr0 = 0.001,
        augment = True,  
        workers = 0
        # seed = current_seed,      # Apply the generated random seed
        # deterministic = False,    # Disable YOLO's enforced reproducibility
    )

if __name__ == "__main__":
    freeze_support() 
    main()
