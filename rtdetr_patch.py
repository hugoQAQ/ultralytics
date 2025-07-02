# rtdetr_patch.py
# Purpose: Provides functionality to monkey patch the RTDETRDecoder from Ultralytics
# to extract raw classification logits and background scores during inference.

import torch

# --- Global Storage for Extracted Data ---
# This dictionary will be populated by the patched forward method.
# It's designed for simplicity in demonstration/debugging. For more robust applications,
# consider attaching data to the model instance or using a more structured callback mechanism.
extracted_rtdetr_data = {
    "logits": None,  # Stores raw logits from the last decoder layer (batch_size, num_queries, num_classes_inc_bg)
    "background_scores": None  # Stores background scores (logits for the background class) (batch_size, num_queries)
}

# --- Original Method Storage ---
# This variable will hold a reference to the original RTDETRDecoder.forward method
# before it's replaced by the patched version.
original_rtdetrdecoder_forward = None

# --- Patched Forward Method ---
def patched_rtdetrdecoder_forward(self, x, batch=None):
    """
    Patched version of ultralytics.nn.modules.head.RTDETRDecoder.forward.

    This wrapper calls the original forward method and then extracts:
    1. Raw classification logits from the final decoder layer.
    2. Scores (logits) specifically for the background class.

    These are stored in the global `extracted_rtdetr_data` dictionary.

    Args:
        self: The instance of RTDETRDecoder.
        x (list): List of feature maps from the backbone.
        batch (dict, optional): Batch information, typically used during training.

    Returns:
        The original output of the RTDETRDecoder.forward method, ensuring
        that the model's normal operation is unaffected.
    """
    global original_rtdetrdecoder_forward, extracted_rtdetr_data

    if original_rtdetrdecoder_forward is None:
        # This should not happen if apply_rtdetr_patch() was called correctly.
        raise RuntimeError("Original RTDETRDecoder.forward method not saved. "
                           "Ensure apply_rtdetr_patch() is called before model inference.")

    # Call the original forward method of RTDETRDecoder
    original_output = original_rtdetrdecoder_forward(self, x, batch)

    # The structure of original_output depends on the mode (training, eval, export).
    # RTDETRDecoder.forward internally calls self.decoder (DeformableTransformerDecoder),
    # which returns dec_bboxes_stacked, dec_scores_stacked.
    # - dec_scores_stacked: (num_layers, bs, num_queries, num_classes_including_bg)
    # These are the raw logits we need.

    dec_scores_from_output = None
    if self.training:
        # In training, original_output is: (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)
        # dec_scores is dec_scores_stacked.
        dec_scores_from_output = original_output[1]
    elif not self.export:  # Evaluation mode, but not when exporting the model
        # In eval (non-export), original_output is: (y, x_tuple)
        # where y is the processed tensor for prediction, and
        # x_tuple is (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta).
        # So, x_tuple[1] (which is original_output[1][1]) is dec_scores_stacked.
        if isinstance(original_output, tuple) and len(original_output) == 2 and \
           isinstance(original_output[1], tuple) and len(original_output[1]) >= 2:
            dec_scores_from_output = original_output[1][1]
    # If self.export is True, original_output is just the final processed tensor 'y'.
    # Extracting raw logits in export mode via this patch is not straightforward
    # as they are processed before being returned. This patch primarily targets
    # non-export use cases (training, evaluation, direct inference in Python).

    if dec_scores_from_output is not None:
        # We are interested in the logits from the last decoder layer.
        # Shape: (batch_size, num_queries, num_classes_including_background)
        raw_logits = dec_scores_from_output[-1].clone().detach()

        # self.nc is the number of foreground classes.
        # The background class is conventionally at index self.nc.
        background_class_index = self.nc

        if raw_logits.shape[-1] <= background_class_index:
            # This could happen if self.nc is miscalculated or num_classes is smaller than expected.
            print(f"Warning: background_class_index ({background_class_index}) is out of bounds "
                  f"for logits_tensor with shape {raw_logits.shape}. Cannot extract background scores.")
            extracted_rtdetr_data["background_scores"] = None
        else:
            # Shape: (batch_size, num_queries)
            background_scores = raw_logits[..., background_class_index].clone().detach()
            extracted_rtdetr_data["background_scores"] = background_scores

        extracted_rtdetr_data["logits"] = raw_logits

    else:
        # This might occur in export mode or if the model's output structure changes.
        extracted_rtdetr_data["logits"] = None
        extracted_rtdetr_data["background_scores"] = None
        # Users should be aware if data isn't being captured.
        print("Warning: Could not extract logits in patched_rtdetrdecoder_forward. "
              "Model might be in export mode or its output structure is unexpected.")

    return original_output

# --- Patch Management Functions ---
def apply_rtdetr_patch():
    """
    Applies the monkey patch to `ultralytics.nn.modules.head.RTDETRDecoder.forward`.

    This function replaces the original `forward` method with the
    `patched_rtdetrdecoder_forward` wrapper. It should be called once before
    any RT-DETR model inference if logits extraction is desired.
    """
    global original_rtdetrdecoder_forward
    try:
        from ultralytics.nn.modules.head import RTDETRDecoder

        # Check if already patched to prevent re-patching
        if hasattr(RTDETRDecoder.forward, '_is_patched_by_jules_rtdetr_logit_extractor'):
            print("RTDETRDecoder.forward is already patched. Skipping.")
            return

        original_rtdetrdecoder_forward = RTDETRDecoder.forward
        RTDETRDecoder.forward = patched_rtdetrdecoder_forward

        # Mark the function as patched
        RTDETRDecoder.forward._is_patched_by_jules_rtdetr_logit_extractor = True

        print("Successfully patched RTDETRDecoder.forward for logit extraction.")
    except ImportError:
        print("Error: Could not import RTDETRDecoder. Ensure Ultralytics is installed and accessible.")
    except Exception as e:
        print(f"An error occurred during patching: {e}")

def remove_rtdetr_patch():
    """
    Removes the monkey patch from `ultralytics.nn.modules.head.RTDETRDecoder.forward`.

    Restores the original `forward` method. Call this if you want to revert
    to the default behavior or clean up after extraction.
    """
    global original_rtdetrdecoder_forward
    if original_rtdetrdecoder_forward is not None:
        try:
            from ultralytics.nn.modules.head import RTDETRDecoder

            if hasattr(RTDETRDecoder.forward, '_is_patched_by_jules_rtdetr_logit_extractor'):
                RTDETRDecoder.forward = original_rtdetrdecoder_forward
                original_rtdetrdecoder_forward = None # Clear the stored original method
                # The marker attribute was on the patched function, which is now replaced.
                # The original function (now restored) won't have this attribute.
                print("Successfully removed RTDETRDecoder.forward patch.")
            else:
                print("Patch not found on RTDETRDecoder.forward or already removed.")
        except ImportError:
            print("Error: Could not import RTDETRDecoder during patch removal.")
        except Exception as e:
            print(f"An error occurred during patch removal: {e}")
    else:
        print("Patch was not applied or already removed (original_rtdetrdecoder_forward is None).")

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # This block demonstrates how to use the patch.
    # In a real scenario, model loading and inference would typically occur
    # after `apply_rtdetr_patch()` and before `remove_rtdetr_patch()`.

    print("Illustrative usage of rtdetr_patch.py:")

    # 1. Apply the patch
    apply_rtdetr_patch()
    # Try applying again to show it handles re-patching attempts
    apply_rtdetr_patch()


    # --- Placeholder for model loading and prediction ---
    # print("\nImagine RT-DETR model loading and prediction occurring here...")
    # # Example:
    # # from ultralytics import RTDETR
    # # model = RTDETR('rtdetr-l.pt')
    # # # The patched forward method will be called during model inference
    # # results = model('path/to/your/image.jpg')
    #
    # # After the above (imagined) prediction, data would be in extracted_rtdetr_data
    # # For this standalone example, we'll manually populate dummy data to show access:
    # extracted_rtdetr_data["logits"] = torch.randn(1, 300, 81) # Batch, Queries, Classes_inc_BG
    # extracted_rtdetr_data["background_scores"] = torch.randn(1, 300) # Batch, Queries
    # --- End of placeholder ---

    print("\nAccessing potentially extracted data (dummy data for this example):")
    if extracted_rtdetr_data["logits"] is not None:
        print(f"  Extracted logits shape: {extracted_rtdetr_data['logits'].shape}")
        print(f"  Extracted background_scores shape: {extracted_rtdetr_data['background_scores'].shape}")
    else:
        print("  No data extracted (or dummy data not populated in this standalone run).")

    # Clean up dummy data for clarity if a real test follows
    extracted_rtdetr_data["logits"] = None
    extracted_rtdetr_data["background_scores"] = None

    # 2. Remove the patch
    print("\nRemoving the patch...")
    remove_rtdetr_patch()
    # Try removing again
    remove_rtdetr_patch()

    print("\nPatching mechanism demonstration complete.")
```

Recreating `test_rtdetr_patch.py`:
