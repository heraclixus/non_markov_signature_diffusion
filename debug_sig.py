import torch
import pysiglib.torch_api as pysiglib
import inspect

print("Checking pysiglib.signature...")
print(f"Type: {type(pysiglib.signature)}")

if hasattr(pysiglib, "Signature"):
    print("\nFound Signature class (assuming it's the autograd function):")
    sig_cls = pysiglib.Signature
    print(f"Forward signature: {inspect.signature(sig_cls.forward)}")
    print(f"Backward signature: {inspect.signature(sig_cls.backward)}")
    
    # Check if we can reproduce the error with a minimal example
    try:
        print("\nAttempting minimal example...")
        path = torch.randn(2, 5, 3, requires_grad=True)
        # Using arguments from train_dart.py: 
        # signature(path, degree=..., time_aug=..., lead_lag=...)
        # Note: pysiglib.signature might be an alias for Signature.apply
        
        sig = pysiglib.signature(path, degree=2, time_aug=False, lead_lag=False)
        print("Forward pass successful.")
        
        loss = sig.sum()
        loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"\nError during minimal example: {e}")

else:
    print("\nCould not find Signature class in pysiglib.torch_api")
    print(dir(pysiglib))

