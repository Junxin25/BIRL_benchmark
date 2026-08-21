import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path("/home/junxinfu/BIRL_benchmark")
PYTHON_BIN = Path("/home/junxinfu/.conda/envs/Deeperhistreg/bin/python")
BASE_CONFIG = ROOT / "configs/deeperhistreg_swint_superglue_nomask_tiled_birl_gpu_flipfalse.json"
TABLE = ROOT / "data-images/pairs-imgs-lnds_anhir_training_available_1.csv"
DATASET = Path("/home/junxinfu/ANHIR_dataset_medium")
OUT_DIR = ROOT / "results"
SCRIPT = Path("/home/junxinfu/registration/Model/DeeperHistReg/deeperhistreg/run.py")
CONFIG_DIR = ROOT / "configs/generated_tre_search"


VARIANTS = [
    {
        "name": "rigid_a60_s1024_m002_max2048",
        "updates": {
            "transform_type": "rigid",
            "use_ransac_transform": False,
            "angle_step": 60,
            "registration_size": 1024,
            "match_threshold": 0.02,
            "max_keypoints": 2048,
            "tile_overlap": 0.25,
        },
    },
    {
        "name": "rigid_a60_s1024_m000_max4096",
        "updates": {
            "transform_type": "rigid",
            "use_ransac_transform": False,
            "angle_step": 60,
            "registration_size": 1024,
            "match_threshold": 0.0,
            "max_keypoints": 4096,
            "tile_overlap": 0.25,
        },
    },
    {
        "name": "affine_a30_s1024_m002_max2048",
        "updates": {
            "transform_type": "affine",
            "use_ransac_transform": False,
            "angle_step": 30,
            "registration_size": 1024,
            "match_threshold": 0.02,
            "max_keypoints": 2048,
            "tile_overlap": 0.25,
        },
    },
    {
        "name": "affine_a60_s1024_m005_max2048",
        "updates": {
            "transform_type": "affine",
            "use_ransac_transform": False,
            "angle_step": 60,
            "registration_size": 1024,
            "match_threshold": 0.05,
            "max_keypoints": 2048,
            "tile_overlap": 0.25,
        },
    },
]


def write_config(name, updates):
    with BASE_CONFIG.open() as fp:
        config = json.load(fp)
    params = config["initial_registration_params"]
    params.update(updates)
    params["save_intermediate"] = True
    params["echo"] = True
    config["run_nonrigid_registration"] = False
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"deeperhistreg_swint_tiled_{name}.json"
    with path.open("w") as fp:
        json.dump(config, fp, indent=2)
    return path


def latest_experiment_dir(prefix):
    candidates = sorted(OUT_DIR.glob(f"BmDeeperHistReg_{prefix}_*"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def read_tre(exp_dir):
    path = exp_dir / "registration-results.csv"
    if not path.exists():
        return None
    with path.open(newline="") as fp:
        row = next(csv.DictReader(fp), None)
    if not row:
        return None
    return float(row["TRE Mean"])


def read_matches(exp_dir):
    path = exp_dir / "0/registration.log"
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    matches = re.findall(r"Final matches:\s*([0-9]+)", text)
    return int(matches[-1]) if matches else None


def main():
    results = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        config_path = write_config(variant["name"], variant["updates"])
        bench_name = f"SwinTTiledTreSearch_{variant['name']}"
        cmd = [
            str(PYTHON_BIN),
            "bm_experiments/bm_DeeperHistReg.py",
            "-t",
            str(TABLE),
            "-d",
            str(DATASET),
            "-o",
            str(OUT_DIR),
            "-py",
            str(PYTHON_BIN),
            "-script",
            str(SCRIPT),
            "-params",
            str(config_path),
            "--name",
            bench_name,
            "--case_name",
            f"BIRL_DeeperHistReg_{variant['name']}",
            "--unique",
            "--nb_workers",
            "1",
        ]
        print(f"\n=== Running {variant['name']} ===", flush=True)
        completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
        exp_dir = latest_experiment_dir(bench_name)
        tre = read_tre(exp_dir) if exp_dir else None
        final_matches = read_matches(exp_dir) if exp_dir else None
        result = {
            "variant": variant["name"],
            "returncode": completed.returncode,
            "tre_mean": tre,
            "final_matches": final_matches,
            "exp_dir": str(exp_dir) if exp_dir else "",
        }
        results.append(result)
        print(
            f"RESULT {variant['name']}: returncode={completed.returncode}, "
            f"TRE Mean={tre}, Final matches={final_matches}, exp={exp_dir}",
            flush=True,
        )

    print("\n=== Sorted by TRE Mean ===")
    for row in sorted(results, key=lambda x: float("inf") if x["tre_mean"] is None else x["tre_mean"]):
        print(
            f"{row['variant']}: TRE Mean={row['tre_mean']}, "
            f"Final matches={row['final_matches']}, returncode={row['returncode']}, "
            f"exp={row['exp_dir']}"
        )


if __name__ == "__main__":
    main()
