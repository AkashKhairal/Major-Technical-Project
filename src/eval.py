import argparse
import os
import subprocess
import time


def evaluate_icdar(gt_path, pred_path):
    """
    Evaluate ICDAR2015 detection results using official evaluation script.
    Assumes prediction files (res_*.txt) already exist.
    """

    assert os.path.exists(gt_path), f"GT path not found: {gt_path}"
    assert os.path.exists(pred_path), f"Prediction path not found: {pred_path}"

    start_time = time.time()

    # Go to predictions directory
    cwd = os.getcwd()
    os.chdir(pred_path)

    # Zip all prediction txt files
    if os.path.exists("submit.zip"):
        os.remove("submit.zip")

    subprocess.run("zip -q submit.zip *.txt", shell=True)

    # Move zip back
    subprocess.run("mv submit.zip ..", shell=True)
    os.chdir(cwd)

    submit_zip = os.path.join(os.path.dirname(pred_path), "submit.zip")

    # Run official ICDAR evaluation
    eval_cmd = (
        f"python evaluate/script.py "
        f"-g={gt_path}/gt.zip "
        f"-s={submit_zip}"
    )

    print("Running ICDAR Evaluation...")
    print(eval_cmd)
    result = subprocess.getoutput(eval_cmd)

    print("\n========== Evaluation Result ==========")
    print(result)
    print("======================================")

    # Cleanup
    os.remove(submit_zip)

    print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser("ICDAR2015 Evaluation")
    parser.add_argument(
        "--gt_path",
        required=True,
        help="Path to ground truth folder (containing gt.zip)",
    )
    parser.add_argument(
        "--pred_path",
        required=True,
        help="Path to prediction folder (containing res_*.txt files)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_icdar(args.gt_path, args.pred_path)
