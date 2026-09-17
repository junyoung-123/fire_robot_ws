"""Keep lazy model setup from overriding the node's CPU thread budget."""


def bounded_yolo_inference(model, image, threads, **kwargs):
    if threads <= 0:
        return model(image, **kwargs)
    import torch
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    try:
        return model(image, **kwargs)
    finally:
        # Ultralytics select_device resets this during first predictor setup.
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
