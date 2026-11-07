from ultralytics import YOLO

def main():
    # 1. Load trained or pre-trained model
    model = YOLO(r"path_to_model")

    # 2. Run validation on your dataset
    metrics = model.val(
        data=r"path_to_dataset",
        split="test" 
    )
    
    # --- Overall Mean Metrics ---
    mAP50_95 = metrics.box.map    
    mAP50    = metrics.box.map50    
    mAP75    = metrics.box.map75   
    mean_p   = metrics.box.mp        
    mean_r   = metrics.box.mr      
    mean_f1  = metrics.box.f1.mean() 

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
        ap = metrics.box.maps[class_idx]  
        
        print(f"\nClass: {class_name}")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  mAP50-95:  {ap:.4f}")

if __name__ == '__main__':
    main()
