import torch
from ultralytics import RTDETR
from ultralytics.utils import ASSETS # For sample images

# Import the patching utilities and the global data dictionary
from rtdetr_patch import apply_rtdetr_patch, remove_rtdetr_patch, extracted_rtdetr_data

def run_test():
    print("--- Testing RT-DETR Monkey Patch ---")

    # --- 1. Apply the patch ---
    print("\nApplying RT-DETR patch...")
    apply_rtdetr_patch()

    # --- 2. Load the model ---
    model_name = 'rtdetr-l.pt'
    print(f"\nLoading RT-DETR model: {model_name}...")
    try:
        model = RTDETR(model_name)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please ensure the model weights (e.g., rtdetr-l.pt) are available.")
        print("You might need to run 'yolo checks' or download them manually if this is the first time.")
        return

    # --- 3. Perform prediction ---
    sample_image = ASSETS / 'bus.jpg' # Using a standard asset
    print(f"\nPerforming prediction on: {sample_image}...")
    try:
        results = model(sample_image, verbose=False)
        print("Prediction complete.")
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    # --- 4. Access and print extracted data ---
    print("\n--- Extracted Data ---")
    if extracted_rtdetr_data["logits"] is not None:
        print("Logits successfully extracted!")
        logits_tensor = extracted_rtdetr_data["logits"]
        print(f"  Shape of logits: {logits_tensor.shape}")
        print(f"  Logits (first query, first 5 classes): {logits_tensor[0, 0, :5]}")

        background_scores_tensor = extracted_rtdetr_data["background_scores"]
        print("\nBackground scores successfully extracted!")
        print(f"  Shape of background_scores: {background_scores_tensor.shape}")
        print(f"  Background scores (first 5 queries): {background_scores_tensor[0, :5]}")

        try:
            num_foreground_classes = model.model.model[-1].nc
            is_match = torch.allclose(background_scores_tensor, logits_tensor[..., num_foreground_classes])
            print(f"\n  Background scores match logits[..., {num_foreground_classes}]? {is_match}")
            if not is_match:
                print("    Mismatch detected. This could indicate an issue with background_class_index logic.")
        except Exception as e:
            print(f"    Could not verify background scores against logits: {e}")
    else:
        print("Data not extracted. Check patch application or model execution mode (e.g., export mode).")

    # --- 5. Check standard model output (optional but good for sanity) ---
    print("\n--- Standard Model Output (Sanity Check) ---")
    if results:
        first_result = results[0]
        print(f"Number of detections in the first image: {len(first_result.boxes)}")
        if len(first_result.boxes) > 0:
            print(f"  Example detection (first box): {first_result.boxes[0].xyxyn.tolist()} conf: {first_result.boxes[0].conf.tolist()} cls: {first_result.boxes[0].cls.tolist()}")
    else:
        print("No results from model prediction.")

    # --- 6. Remove the patch and test again (optional) ---
    print("\nRemoving RT-DETR patch...")
    remove_rtdetr_patch()

    extracted_rtdetr_data["logits"] = None
    extracted_rtdetr_data["background_scores"] = None

    print("\nPerforming prediction with patch removed...")
    try:
        results_unpatched = model(sample_image, verbose=False)
        print("Prediction with removed patch complete.")
    except Exception as e:
        print(f"Prediction failed after patch removal: {e}")
        return

    print("\n--- Extracted Data (after patch removal) ---")
    if extracted_rtdetr_data["logits"] is not None or extracted_rtdetr_data["background_scores"] is not None:
        print("Error: Data was extracted even after removing the patch.")
        if extracted_rtdetr_data["logits"] is not None:
            print(f"  Logits shape: {extracted_rtdetr_data['logits'].shape}")
        if extracted_rtdetr_data["background_scores"] is not None:
            print(f"  Background_scores shape: {extracted_rtdetr_data['background_scores'].shape}")
    else:
        print("No data extracted after patch removal (as expected).")

    print("\n--- Test Complete ---")

if __name__ == '__main__':
    run_test()
```

Now, running the test script:
