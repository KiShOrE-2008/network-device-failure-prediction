import os
import sys
import subprocess

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    
    venv_subpath = os.path.join("Scripts", "python.exe") if os.name == "nt" else os.path.join("bin", "python")
    candidates = [
        os.path.join(root_dir, "venv", venv_subpath),
        os.path.join(root_dir, ".venv", venv_subpath),
    ]
    python_bin = next((c for c in candidates if os.path.exists(c)), sys.executable)

    backend_app = os.path.join(backend_dir, "app.py")
    cmd = [python_bin, backend_app] + sys.argv[1:]
    
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "2")
    env.setdefault("NUMEXPR_NUM_THREADS", "2")

    try:
        subprocess.run(cmd, cwd=backend_dir, env=env)
    except KeyboardInterrupt:
        print("\n🛑 Execution stopped by user.")

if __name__ == "__main__":
    main()
