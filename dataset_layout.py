"""Small, explicit dataset-layout discovery helpers.

The public interfaces accept either case-first or type-first layouts.  This
module only discovers paths; it does not read or modify any data.
"""

from __future__ import annotations

from pathlib import Path


MASK_DIR_NAMES = ("mask", "masks")
MESH_DIR_NAMES = ("mesh", "meshes")


def _single_directory(parent: Path, names: tuple[str, ...], kind: str) -> Path | None:
    found = [parent / name for name in names if (parent / name).is_dir()]
    if len(found) > 1:
        raise ValueError(
            f"Ambiguous {kind} directories under {parent}: "
            + ", ".join(str(path) for path in found)
        )
    return found[0] if found else None


def discover_inference_masks(input_root: Path) -> dict[str, Path]:
    """Discover ``case -> mask directory`` in either public input layout."""

    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    discovered: dict[str, Path] = {}

    # Case-first: input/<case>/(mask|masks)/<frame>.nii.gz
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        mask_dir = _single_directory(case_dir, MASK_DIR_NAMES, "mask")
        if mask_dir is not None:
            discovered[case_dir.name] = mask_dir

    # Type-first: input/(mask|masks)/<case>/<frame>.nii.gz
    mask_root = _single_directory(root, MASK_DIR_NAMES, "mask")
    if mask_root is not None:
        for case_dir in sorted(path for path in mask_root.iterdir() if path.is_dir()):
            previous = discovered.get(case_dir.name)
            if previous is not None and previous.resolve() != case_dir.resolve():
                raise ValueError(
                    f"Case {case_dir.name} appears in both inference layouts: "
                    f"{previous} and {case_dir}"
                )
            discovered[case_dir.name] = case_dir
    return discovered


def discover_training_cases(dataset_root: Path) -> dict[str, dict[str, Path]]:
    """Discover VTK cases; a missing transform is allowed when Masks exist."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    discovered: dict[str, dict[str, Path]] = {}

    # Case-first: <case>/mesh|meshes, plus P.txt or mask(s) for generating it.
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        mesh_dir = _single_directory(case_dir, MESH_DIR_NAMES, "mesh")
        transform = case_dir / "P.txt"
        if mesh_dir is None and not transform.exists():
            continue
        if mesh_dir is None:
            raise ValueError(
                f"Incomplete case-first training layout for {case_dir.name}: "
                f"expected {case_dir}/mesh(or meshes)"
            )
        mask_dir = _single_directory(case_dir, MASK_DIR_NAMES, "mask")
        if not transform.is_file() and mask_dir is None:
            raise ValueError(
                f"Case {case_dir.name} needs either {transform} or a mask(s) "
                "directory from which it can be generated"
            )
        discovered[case_dir.name] = {
            "mesh_dir": mesh_dir,
            "transform": transform,
            **({"mask_dir": mask_dir} if mask_dir is not None else {}),
        }

    # Type-first: mesh|meshes/<case>, plus P/<case>.txt or mask(s)/<case>.
    mesh_root = _single_directory(root, MESH_DIR_NAMES, "mesh")
    transform_root = root / "P"
    if mesh_root is not None or transform_root.exists():
        if mesh_root is None:
            raise ValueError(
                f"Incomplete type-first training layout under {root}: "
                "expected mesh(or meshes)"
            )
        mask_root = _single_directory(root, MASK_DIR_NAMES, "mask")
        type_cases = {
            path.name: path
            for path in mesh_root.iterdir()
            if path.is_dir()
        }
        for case_id, mesh_dir in sorted(type_cases.items()):
            transform = transform_root / f"{case_id}.txt"
            mask_dir = (
                mask_root / case_id
                if mask_root is not None and (mask_root / case_id).is_dir()
                else None
            )
            if not transform.is_file() and mask_dir is None:
                raise ValueError(
                    f"Case {case_id} needs either {transform} or a mask(s) "
                    "directory from which it can be generated"
                )
            previous = discovered.get(case_id)
            if previous is not None:
                raise ValueError(
                    f"Case {case_id} appears in both training layouts: "
                    f"{previous['mesh_dir']} and {mesh_dir}"
                )
            item = {"mesh_dir": mesh_dir, "transform": transform}
            if mask_dir is not None:
                item["mask_dir"] = mask_dir
            discovered[case_id] = item
    return discovered
