import sys, importlib.util
spec = importlib.util.spec_from_file_location("demo", "examples/demo_deepseek.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print("Module loaded OK")
except SystemExit:
    print("Module loaded but called sys.exit")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
