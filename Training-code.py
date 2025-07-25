from ultralytics import YOLO
from multiprocessing import freeze_support
import os

def main():

    model = YOLO("yolov8m.pt")

    CWD = os.getcwd()
    Yaml = os.path.join(CWD, 'MyDataSet', 'Modified.yaml')

    results = model.train(
        data = Yaml,
        epochs = 30,
        imgsz = 256,
        batch = 8,
        device = 'cuda',
        optimizer ="Adam",   
        name = "sign_language_yolo",
        lr0 = 0.001,
        augment = True,  
        workers = 0,
    )

if __name__ == "__main__":
    freeze_support() 
    main()
