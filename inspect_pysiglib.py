import inspect
import os
import torch

try:
    import pysiglib
    import pysiglib.torch_api
    print(f"pysiglib imported from: {pysiglib.__file__}")
except ImportError:
    print("pysiglib not found")
    exit(1)

func = pysiglib.torch_api.signature
print(f"pysiglib.signature is: {func}")

# Try to get source
try:
    print("\n--- Source of pysiglib.torch_api.signature ---")
    print(inspect.getsource(func))
except Exception as e:
    print(f"Could not get source: {e}")

# Inspect closure to find the Autograd Function
sig_cls = None
if hasattr(func, "__closure__") and func.__closure__:
    print("\n--- Inspecting closure ---")
    for i, cell in enumerate(func.__closure__):
        content = cell.cell_contents
        print(f"Cell {i}: {type(content)}")
        if isinstance(content, type) and issubclass(content, torch.autograd.Function):
            print(f"  FOUND AUTOGRAD FUNCTION: {content.__name__}")
            sig_cls = content

if sig_cls:
    print(f"\n--- Inspecting {sig_cls.__name__} ---")
    try:
        fwd_sig = inspect.signature(sig_cls.forward)
        print(f"Forward signature: {fwd_sig}")
    except Exception as e:
        print(f"Could not inspect forward signature: {e}")
    
    # Check strict argument count if possible
    # backward depends on how many arguments forward *received* usually? 
    # Or for static backward, it corresponds to inputs.
    
    # Try to verify the fix: wrap it?
    pass

print("\n--- End Inspection ---")
