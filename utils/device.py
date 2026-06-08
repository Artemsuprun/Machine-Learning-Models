import torch


def device_info():
    # Base Device Setup
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    
    device = torch.device(device_type)
    print(f"=== PyTorch Device Info ===")
    print(f"Active Backend: {device_type.upper()}\n")

    # Hardware Specific Details
    if device.type == "cuda":
        print(f"Number of GPUs Available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"--- GPU {i} ---")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-Processors: {props.multi_processor_count}")
    elif device.type == "mps":
        print("Metal Performance Shaders (MPS) is available.")
        print("Note: Apple silicon memory is unified with system memory.")
    else:
        print("Running on CPU.")


def get_device():
    device = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    return device

if __name__ == "__main__":
    device_info()