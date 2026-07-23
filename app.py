import subprocess
import os
import sys

def main():
    # Detect the path to the virtual environment python
    # Works on Linux/macOS ("venv/bin/python") and Windows ("venv/Scripts/python.exe")
    if os.name == "nt":
        python_bin = os.path.join("venv", "Scripts", "python.exe")
    else:
        python_bin = os.path.join("venv", "bin", "python")

    # If the venv python is not found, fallback to the current python runner
    if not os.path.exists(python_bin):
        print(f"⚠️ Virtual environment python not found at '{python_bin}'.")
        print("Falling back to the current python interpreter.")
        python_bin = sys.executable

    # Define the execution pipeline in order
    predict_args = sys.argv[1:]
    pipeline = [
        ("src/generate_dataset.py", []),
        ("src/eda.py", []),
        ("src/train_model.py", []),
        ("src/predict.py", predict_args)
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
            subprocess.run(cmd, check=True)
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
