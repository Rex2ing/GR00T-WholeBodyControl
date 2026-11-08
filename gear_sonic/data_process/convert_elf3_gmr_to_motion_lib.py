#!/usr/bin/env python3
"""Convert filtered ELF3 GMR PKLs to SONIC robot motions, without SMPL data.

Run with the SONIC Python environment (NumPy 1.x) so its joblib reader can load
the output. Input NumPy 2 pickle module names are adapted only while reading.
The input tree is read-only; every output has a unique, path-derived motion ID.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import tempfile

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/elf3.xml"


class GMRUnpickler(pickle.Unpickler):
    """Read GMR NumPy 2 arrays using the SONIC NumPy 1.x environment."""

    def find_class(self, module, name):
        if module.startswith("numpy._core") and int(np.__version__.split(".")[0]) < 2:
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def read_gmr(path):
    raw = Path(path).read_bytes()
    return GMRUnpickler(io.BytesIO(raw)).load(), hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def motion_id(relative_path):
    relative = Path(relative_path).as_posix()
    readable = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(Path(relative).with_suffix("")))[:130]
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
    return f"amass__{readable}__{digest}"


def hinge_info(model):
    joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in joints]
    if model.nq != 7 + len(joints) or model.nv != 6 + len(joints):
        raise ValueError("Expected one free root and hinge-only actuated joints")
    if not np.array_equal(model.jnt_qposadr[joints], np.arange(7, model.nq)):
        raise ValueError("Hinge columns must follow the seven root qpos values")
    return joints, names


class ELF3Converter:
    def __init__(self, target_xml, source_xml):
        self.target_xml = Path(target_xml).resolve()
        self.source_xml = Path(source_xml).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.target_xml))
        source_model = mujoco.MjModel.from_xml_path(str(self.source_xml))
        self.joints, self.joint_names = hinge_info(self.model)
        source_joints, source_names = hinge_info(source_model)
        if len(self.joints) != 29 or self.model.nbody != 31:
            raise ValueError("ELF3 output requires 29 hinge DOFs and 30 bodies excluding world")
        if len(set(source_names)) != 29 or set(source_names) != set(self.joint_names):
            raise ValueError("Source and target models must have the same 29 joint names")
        self.source_columns = np.array([source_names.index(n) for n in self.joint_names])
        source_ids = source_joints[self.source_columns]
        for field in ("jnt_axis", "jnt_pos", "jnt_range"):
            if not np.allclose(getattr(source_model, field)[source_ids],
                               getattr(self.model, field)[self.joints], atol=1e-10, rtol=0):
                raise ValueError(f"Source/target kinematic mismatch: {field}")
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                           for i in range(1, self.model.nbody)]
        for target_id, name in enumerate(self.body_names, start=1):
            source_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if source_id < 0:
                raise ValueError(f"Source model has no body {name}")
            for field in ("body_pos", "body_quat"):
                if not np.allclose(getattr(source_model, field)[source_id],
                                   getattr(self.model, field)[target_id], atol=1e-10, rtol=0):
                    raise ValueError(f"Source/target {field} mismatch for {name}")
            old_parent = mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_BODY,
                                         int(source_model.body_parentid[source_id]))
            new_parent = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                         int(self.model.body_parentid[target_id]))
            if old_parent != new_parent:
                raise ValueError(f"Source/target parent mismatch for {name}")
        self.body_slots = self.model.jnt_bodyid[self.joints] - 1
        if set(self.body_slots) != set(range(1, 30)):
            raise ValueError("Expected one hinge on every non-root ELF3 body")
        self.axes = self.model.jnt_axis[self.joints]
        # Humanoid_Batch recovers hinge angles by summing pose_aa's XYZ components.
        if not np.all(np.isin(self.axes, [0, 1])) or not np.all(self.axes.sum(axis=1) == 1):
            raise ValueError("Current SONIC FK requires positive canonical ELF3 hinge axes")
        self.data = mujoco.MjData(self.model)

    def convert(self, raw):
        if not isinstance(raw, dict):
            raise ValueError("GMR input must be a flat motion dictionary")
        pos, quat, angles = [np.asarray(raw[k], dtype=np.float64)
                             for k in ("root_pos", "root_rot", "dof_pos")]
        fps = float(raw["fps"])
        frames = len(pos)
        if frames < 2 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Expected at least two frames and a positive finite FPS")
        for name, array, width in (("root_pos", pos, 3), ("root_rot", quat, 4),
                                   ("dof_pos", angles, 29)):
            if array.shape != (frames, width) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name}: expected finite ({frames}, {width})")
        if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-5, rtol=0):
            raise ValueError("Input root_rot must be normalized xyzw quaternions")
        angles = angles[:, self.source_columns]
        pose = np.zeros((frames, 30, 3), dtype=np.float32)
        pose[:, 0] = Rotation.from_quat(quat).as_rotvec()
        pose[:, self.body_slots] = self.axes[None] * angles[..., None]
        entry = {
            "root_trans_offset": pos.astype(np.float32),
            "pose_aa": pose,
            "dof": angles.astype(np.float32),
            "root_rot": quat[:, [3, 0, 1, 2]].astype(np.float32),
            "fps": int(fps) if fps.is_integer() else fps,
        }
        errors = {
            "max_translation_error_m": float(np.max(np.abs(entry["root_trans_offset"] - pos))),
            "max_dof_error_rad": float(np.max(np.abs(entry["dof"] - angles))),
            "max_root_rotation_error_rad": float(np.max((
                Rotation.from_rotvec(pose[:, 0]).inv() * Rotation.from_quat(quat)
            ).magnitude())),
            "max_cached_fk_error_m": self.check_cached_fk(raw, entry),
        }
        lower, upper = self.model.jnt_range[self.joints].T
        errors["max_joint_limit_excess_rad"] = float(np.maximum(
            np.maximum(lower - angles, angles - upper), 0
        ).max())
        if errors["max_joint_limit_excess_rad"] > 0.020001:
            raise ValueError("Joint limit excess exceeds the source filter's 0.02 rad tolerance")
        if errors["max_root_rotation_error_rad"] > 1e-6:
            raise ValueError("Root axis-angle conversion failed its rotation round trip")
        return entry, errors

    def check_cached_fk(self, raw, entry):
        if raw.get("local_body_pos") is None:
            return None
        names = raw["link_body_list"]
        if len(set(names)) != len(names):
            raise ValueError("Input link_body_list has duplicate names")
        cache = np.asarray(raw["local_body_pos"])
        if cache.shape != (len(entry["dof"]), len(names), 3) or not np.isfinite(cache).all():
            raise ValueError("Invalid cached local_body_pos")
        body_columns = [names.index(n) for n in self.body_names]
        maximum = 0.0
        for frame in sorted({0, len(cache) // 2, len(cache) - 1}):
            self.data.qpos[:3] = 0
            self.data.qpos[3:7] = [1, 0, 0, 0]
            self.data.qpos[7:] = entry["dof"][frame]
            mujoco.mj_kinematics(self.model, self.data)
            maximum = max(maximum, float(np.abs(
                self.data.xpos[1:] - cache[frame, body_columns]
            ).max()))
        if maximum > 1e-4:
            raise ValueError(f"Input cached FK differs from the target model by {maximum} m")
        return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-model", type=Path,
                        help="Defaults to robot_xml from INPUT/_filter_report/run.json")
    parser.add_argument("--limit", type=int, help="Convert the first N files for a smoke test")
    parser.add_argument("--compress", type=int, default=3, choices=range(10))
    args = parser.parse_args()
    source_root, output_root = args.input.resolve(), args.output.resolve()
    if not source_root.is_dir() or source_root == output_root or source_root in output_root.parents:
        parser.error("Input must be a directory and output must be outside the input tree")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        parser.error("Output directory must be empty; choose a fresh output directory")
    files = sorted(source_root.rglob("*.pkl"))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        parser.error("No input PKL files found")
    source_model = args.source_model
    if source_model is None:
        source_model = Path(json.loads((source_root / "_filter_report/run.json").read_text())["robot_xml"])
    converter = ELF3Converter(args.model, source_model)
    identifiers = [motion_id(p.relative_to(source_root)) for p in files]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Motion ID collision")
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "sonic_elf3_motion_lib_v1", "state": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root), "output_root": str(output_root),
        "source_model": str(converter.source_xml), "model": str(converter.target_xml),
        "source_model_sha256": sha256_file(converter.source_xml),
        "model_sha256": sha256_file(converter.target_xml),
        "joint_names_mujoco": converter.joint_names, "body_names_mujoco": converter.body_names,
        "input_root_rot_order": "xyzw", "output_root_rot_order": "wxyz",
        "numpy_version": np.__version__, "joblib_version": joblib.__version__,
        "smpl_generated": False, "mirrors_generated": False,
        "input_files": len(files), "output_files": 0, "frames": 0, "seconds": 0.0,
        "basename_collisions_avoided": len(files) - len({p.stem for p in files}),
        "max_errors": {},
    }
    records = []
    fps_counts, datasets = Counter(), Counter()
    try:
        for index, (source_path, key) in enumerate(zip(files, identifiers), start=1):
            raw, source_hash = read_gmr(source_path)
            entry, errors = converter.convert(raw)
            relative = source_path.relative_to(source_root)
            destination = output_root / relative.parts[0] / f"{key}.pkl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    joblib.dump({key: entry}, stream, compress=args.compress, protocol=4)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(destination)
            frames = len(entry["dof"])
            records.append({
                "motion_id": key, "input_relative_path": relative.as_posix(),
                "output_relative_path": destination.relative_to(output_root).as_posix(),
                "source_sha256": source_hash, "output_sha256": sha256_file(destination),
                "frames": frames, "fps": entry["fps"], "errors": errors,
                "provenance": {k: raw[k] for k in ("source", "filtering", "retarget_profile",
                               "actual_human_height", "foot_grounding", "arm_collision") if k in raw},
            })
            report["output_files"] += 1
            report["frames"] += frames
            report["seconds"] += frames / entry["fps"]
            fps_counts[str(entry["fps"])] += 1
            datasets[relative.parts[0]] += 1
            for name, value in errors.items():
                if value is not None:
                    report["max_errors"][name] = max(report["max_errors"].get(name, 0.0), value)
            if index % 250 == 0 or index == len(files):
                print(f"Converted {index}/{len(files)} motions", flush=True)
        manifest = output_root / "manifest.jsonl"
        with manifest.open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        report["fps_counts"] = dict(fps_counts)
        report["datasets"] = dict(datasets)
        report["manifest_sha256"] = sha256_file(manifest)
        report["state"] = "complete"
    except BaseException as exc:
        report["state"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output_root / "conversion_report.json", report)
    print(json.dumps({k: report[k] for k in ("state", "output_files", "frames", "seconds", "max_errors")}),
          flush=True)


if __name__ == "__main__":
    main()
