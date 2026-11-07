from ultralytics import YOLO

def main():
    # 1. Load trained or pre-trained model
    model = YOLO(r"C:\Main\Coding\Code\GP\runs\detect\5m\weights\best.pt")

    # 2. Run validation on your dataset
# Pass split="test" to evaluate on the test dataset
    metrics = model.val(
        data=r"C:\Main\Coding\Code\GP\MyDataSet\Modified.yaml",
        split="test" 
    )
    
    # --- Overall Mean Metrics ---
    mAP50_95 = metrics.box.map       # mAP@0.5:0.95
    mAP50    = metrics.box.map50     # mAP@0.50
    mAP75    = metrics.box.map75     # mAP@0.75
    mean_p   = metrics.box.mp        # Mean Precision across all classes
    mean_r   = metrics.box.mr        # Mean Recall across all classes
    mean_f1  = metrics.box.f1.mean() # Mean F1-Score across all classes

    print(f"Mean Precision: {mean_p:.4f}")
    print(f"Mean Recall:    {mean_r:.4f}")
    print(f"Mean F1 Score:  {mean_f1:.4f}")
    print(f"mAP@50:         {mAP50:.4f}")
    print(f"mAP@50-95:      {mAP50_95:.4f}")

    # --- Per-Class Metrics ---
    class_names = metrics.names
    for class_idx, class_name in class_names.items():
        p  = metrics.box.p[class_idx]
        r  = metrics.box.r[class_idx]
        f1 = metrics.box.f1[class_idx]
        ap = metrics.box.maps[class_idx]  # class mAP@0.5:0.95
        
        print(f"\nClass: {class_name}")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  mAP50-95:  {ap:.4f}")

# This is the required Windows multiprocessing guard
if __name__ == '__main__':
    main()