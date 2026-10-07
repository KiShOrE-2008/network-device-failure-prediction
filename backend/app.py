import subprocess
import os
import sys

def main():
    backend_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_root)

    # Detect the path to the virtual environment python
    # Works on Linux/macOS ("venv/bin/python") and Windows ("venv/Scripts/python.exe")
    venv_subpath = os.path.join("Scripts", "python.exe") if os.name == "nt" else os.path.join("bin", "python")

    candidates = [
        os.path.join(project_root, "venv", venv_subpath),
        os.path.join(backend_root, "venv", venv_subpath),
        os.path.join(project_root, ".venv", venv_subpath),
        os.path.join(backend_root, ".venv", venv_subpath),
    ]

    python_bin = next((cand for cand in candidates if os.path.exists(cand)), None)

    # If the venv python is not found, fallback to the current python runner
    if not python_bin:
        print("⚠️ Virtual environment python not found in project.")
        print("Falling back to the current python interpreter.")
        python_bin = sys.executable
    else:
        python_bin = os.path.abspath(python_bin)
    # Environment variables to optimize performance
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "2")
    env.setdefault("NUMEXPR_NUM_THREADS", "2")

    # Check if user specifically requested running the CLI pipeline instead of the web server
    run_pipeline = any(arg in sys.argv for arg in ("--pipeline", "--cli", "--train"))

    if not run_pipeline:
        web_app_script = os.path.join(backend_root, "src", "web_app.py")
        cmd = [python_bin, web_app_script]
        print("=" * 60)
        print("🚀 STARTING NETGUARD NOC LOCAL WEB SERVER")
        print("=" * 60)
        print("🌐 Open in your browser: http://localhost:5000")
        print("💡 (To run the CLI ML pipeline instead: python app.py --pipeline)")
        print("=" * 60 + "\n")
        try:
            subprocess.run(cmd, cwd=backend_root, env=env)
        except KeyboardInterrupt:
            print("\n🛑 Web app server stopped by user.")
        sys.exit(0)

    # Define the execution pipeline in order
    predict_args = [arg for arg in sys.argv[1:] if arg not in ("--pipeline", "--cli", "--train")]
    pipeline = [
        (os.path.join(backend_root, "src", "generate_dataset.py"), []),
        (os.path.join(backend_root, "src", "eda.py"), []),
        (os.path.join(backend_root, "src", "train_model.py"), []),
        (os.path.join(backend_root, "src", "predict.py"), predict_args)
    ]

    print("=" * 60)
    print("RUNNING THE NETWORK DEVICE FAILURE PREDICTION PIPELINE")
    print("=" * 60)

    for script, args in pipeline:
        cmd = [python_bin, script] + args
        print(f"\n[Running Step] {' '.join(cmd)}")
        print("-" * 60)
        
        try:
            # Run the command, inheriting stdin, stdout, and stderr so live progress
            # is printed and the interactive prompts in predict.py work correctly.
            subprocess.run(cmd, cwd=backend_root, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Step failed: {script} returned non-zero exit code.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n🛑 Pipeline execution interrupted by user.")
            sys.exit(0)

    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
