import sys
import os

print("Current working directory:", os.getcwd())
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("sys.path:", sys.path)

try:
    import nmsd
    print("\nSuccessfully imported nmsd")
    print("nmsd file:", getattr(nmsd, '__file__', 'unknown'))
    print("nmsd path:", getattr(nmsd, '__path__', 'unknown'))
    
    # Debug directory contents
    if hasattr(nmsd, '__file__'):
        nmsd_dir = os.path.dirname(nmsd.__file__)
        print(f"Contents of {nmsd_dir}:", os.listdir(nmsd_dir))
        data_dir = os.path.join(nmsd_dir, "data")
        if os.path.exists(data_dir):
            print(f"Contents of {data_dir}:", os.listdir(data_dir))
        else:
            print(f"Directory {data_dir} does not exist")
except ImportError as e:
    print("\nFailed to import nmsd:", e)

try:
    import nmsd.data
    print("\nSuccessfully imported nmsd.data")
    print("nmsd.data file:", getattr(nmsd.data, '__file__', 'unknown'))
    print("nmsd.data path:", getattr(nmsd.data, '__path__', 'unknown'))
except ImportError as e:
    print("\nFailed to import nmsd.data:", e)

try:
    from nmsd.data.datasets import get_dataloaders
    print("\nSuccessfully imported get_dataloaders")
except ImportError as e:
    print("\nFailed to import get_dataloaders:", e)

