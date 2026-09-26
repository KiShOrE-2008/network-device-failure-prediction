import os
import sys
import subprocess

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    
    if os.name == "nt":
        python_bin = os.path.join(root_dir, "venv", "Scripts", "python.exe")
    else:
        python_bin = os.path.join(root_dir, "venv", "bin", "python")

    if not os.path.exists(python_bin):
        python_bin = sys.executable

    backend_app = os.path.join(backend_dir, "app.py")
    cmd = [python_bin, backend_app] + sys.argv[1:]
    
    try:
        subprocess.run(cmd, cwd=backend_dir)
    except KeyboardInterrupt:
        print("\n🛑 Execution stopped by user.")

if __name__ == "__main__":
    main()
