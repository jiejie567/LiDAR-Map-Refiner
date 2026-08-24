#!/usr/bin/env python3
"""
Merge all PCD files in a directory into one point cloud.
"""

from __future__ import annotations

import argparse
import io
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_TYPE_MAP = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
    ("I", 8): np.int64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("U", 8): np.uint64,
}

_CANONICAL_FIELDS: Tuple[str, ...] = ("x", "y", "z", "intensity")
_DEFAULT_VIEWPOINT: Tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
_IGNORED_FIELDS: Tuple[str, ...] = ("rgb",)


class PCDCloud:
    def __init__(
        self,
        *,
        fields: Sequence[str],
        sizes: Sequence[int],
        types: Sequence[str],
        counts: Sequence[int],
        data_type: str,
        version: str,
        viewpoint: Sequence[float],
        comments: Sequence[str],
        data: np.ndarray,
        height: int = 1,
    ) -> None:
        self.fields: Tuple[str, ...] = tuple(fields)
        self.sizes: Tuple[int, ...] = tuple(sizes)
        self.types: Tuple[str, ...] = tuple(types)
        self.counts: Tuple[int, ...] = tuple(counts)
        self.data_type = data_type
        self.version = version
        self.viewpoint: Tuple[float, ...] = tuple(viewpoint)
        self.comments: Tuple[str, ...] = tuple(comments)
        self.data = data
        self.height = height

    @property
    def num_points(self) -> int:
        return int(self.data.shape[0])

    def with_data(self, data: np.ndarray, *, height: Optional[int] = None) -> "PCDCloud":
        return PCDCloud(
            fields=self.fields,
            sizes=self.sizes,
            types=self.types,
            counts=self.counts,
            data_type=self.data_type,
            version=self.version,
            viewpoint=self.viewpoint,
            comments=self.comments,
            data=data,
            height=self.height if height is None else height,
        )


def _strip_ignored_fields(cloud: PCDCloud) -> PCDCloud:
    ignored = {name.lower() for name in _IGNORED_FIELDS}
    keep_indices = [
        idx
        for idx, name in enumerate(cloud.fields)
        if name.lower() not in ignored and not name.lower().startswith("unnamed")
    ]
    if len(keep_indices) == len(cloud.fields):
        return cloud
    keep_fields = [cloud.fields[idx] for idx in keep_indices]
    filtered_data = cloud.data[keep_fields].copy()
    return PCDCloud(
        fields=tuple(keep_fields),
        sizes=tuple(cloud.sizes[idx] for idx in keep_indices),
        types=tuple(cloud.types[idx] for idx in keep_indices),
        counts=tuple(cloud.counts[idx] for idx in keep_indices),
        data_type=cloud.data_type,
        version=cloud.version,
        viewpoint=cloud.viewpoint,
        comments=cloud.comments,
        data=filtered_data,
        height=cloud.height,
    )


def _ensure_unique_field_names(fields: Sequence[str], source: Path) -> Tuple[str, ...]:
    sanitized: List[str] = []
    used: set[str] = set()
    counts: Dict[str, int] = {}
    renamed: List[str] = []

    for idx, raw_name in enumerate(fields):
        name = (raw_name or "").strip()
        if not name or all(ch == "_" for ch in name):
            name = "unnamed"
        base = name
        counter = counts.get(base, 0)
        candidate = base if counter == 0 else f"{base}_{counter}"
        while candidate in used:
            counter += 1
            candidate = f"{base}_{counter}"
        counts[base] = counter + 1
        used.add(candidate)
        sanitized.append(candidate)
        if candidate != raw_name:
            renamed.append(f"{raw_name}->{candidate}")

    if renamed:
        changes = ", ".join(renamed)
        print(f"[merge_pcds] Renamed duplicate/empty fields in {source}: {changes}")
    return tuple(sanitized)


def _build_dtype(fields: Sequence[str], sizes: Sequence[int], types: Sequence[str], counts: Sequence[int]) -> np.dtype:
    dtype_fields: List[Tuple[str, object]] = []
    for name, size, typ, count in zip(fields, sizes, types, counts):
        key = (typ.upper(), size)
        if key not in _TYPE_MAP:
            raise ValueError(f"Unsupported PCD field type/size combination: {typ}{size} for field '{name}'")
        base_dtype = _TYPE_MAP[key]
        if count == 1:
            dtype_fields.append((name, base_dtype))
        else:
            dtype_fields.append((name, base_dtype, (count,)))
    return np.dtype(dtype_fields)


def _parse_header(header_lines: Iterable[str]) -> Tuple[Dict[str, List[str]], Tuple[str, ...]]:
    metadata: Dict[str, List[str]] = {}
    comments: List[str] = []
    for line in header_lines:
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
            continue
        parts = line.split()
        key = parts[0].upper()
        metadata[key] = parts[1:]
    return metadata, tuple(comments)


def _structured_to_columns(data: np.ndarray, fields: Sequence[str], counts: Sequence[int]) -> np.ndarray:
    if data.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    columns: List[np.ndarray] = []
    for name, count in zip(fields, counts):
        values = data[name]
        if count == 1:
            columns.append(values.reshape(-1, 1))
        else:
            columns.append(values.reshape(-1, count))
    return np.hstack(columns)


def _column_formats(data: np.ndarray, fields: Sequence[str], counts: Sequence[int]) -> List[str]:
    fmts: List[str] = []
    for name, count in zip(fields, counts):
        subarray = data[name]
        dtype = subarray.dtype
        if np.issubdtype(dtype, np.floating):
            fmt = "%.10f"
        elif np.issubdtype(dtype, np.integer):
            fmt = "%d"
        else:
            raise ValueError(f"Unsupported dtype '{dtype}' for ASCII PCD writing.")
        fmts.extend([fmt] * count)
    return fmts


def _viewpoint_is_default(viewpoint: Sequence[float]) -> bool:
    return np.allclose(viewpoint, _DEFAULT_VIEWPOINT, atol=1e-9)


def _quaternion_to_rotation(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = quat
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _apply_viewpoint_transform(data: np.ndarray, fields: Sequence[str], viewpoint: Sequence[float], source: Path) -> Tuple[np.ndarray, Tuple[float, ...]]:
    if not {"x", "y", "z"}.issubset(fields):
        return data, tuple(viewpoint)
    if _viewpoint_is_default(viewpoint):
        return data, tuple(viewpoint)

    print(f"[merge_pcds] Applying VIEWPOINT transform from {source}")
    points = np.vstack((data["x"], data["y"], data["z"])).T.astype(np.float64, copy=False)
    translation = np.array(viewpoint[:3], dtype=np.float64)
    rotation = _quaternion_to_rotation(viewpoint[3:])
    transformed = points @ rotation.T + translation

    data = data.copy()
    data["x"] = transformed[:, 0].astype(data["x"].dtype, copy=False)
    data["y"] = transformed[:, 1].astype(data["y"].dtype, copy=False)
    data["z"] = transformed[:, 2].astype(data["z"].dtype, copy=False)
    return data, _DEFAULT_VIEWPOINT


def _canonicalize_field_order(cloud: PCDCloud) -> PCDCloud:
    desired_order: List[str] = []
    existing_fields = list(cloud.fields)
    for name in _CANONICAL_FIELDS:
        if name in cloud.fields:
            desired_order.append(name)
    for name in existing_fields:
        if name not in desired_order:
            desired_order.append(name)
    if tuple(desired_order) == cloud.fields:
        return cloud

    index_map = {name: idx for idx, name in enumerate(cloud.fields)}
    new_sizes = [cloud.sizes[index_map[name]] for name in desired_order]
    new_types = [cloud.types[index_map[name]] for name in desired_order]
    new_counts = [cloud.counts[index_map[name]] for name in desired_order]

    reordered_data = cloud.data[desired_order].copy()

    return PCDCloud(
        fields=tuple(desired_order),
        sizes=tuple(new_sizes),
        types=tuple(new_types),
        counts=tuple(new_counts),
        data_type=cloud.data_type,
        version=cloud.version,
        viewpoint=cloud.viewpoint,
        comments=cloud.comments,
        data=reordered_data,
        height=cloud.height,
    )


def _lzf_decompress(payload: bytes, expected_size: int, source: Path) -> bytes:
    output = bytearray()
    src = 0
    payload_len = len(payload)

    while src < payload_len:
        ctrl = payload[src]
        src += 1
        if ctrl < 32:
            literal_len = ctrl + 1
            end = src + literal_len
            if end > payload_len:
                raise ValueError(f"Malformed LZF literal run in {source}")
            output.extend(payload[src:end])
            src = end
            continue

        backref_len = ctrl >> 5
        ref_offset = (ctrl & 0x1F) << 8
        if backref_len == 7:
            if src >= payload_len:
                raise ValueError(f"Malformed LZF extended back-reference in {source}")
            backref_len += payload[src]
            src += 1
        if src >= payload_len:
            raise ValueError(f"Malformed LZF back-reference in {source}")
        ref_offset += payload[src] + 1
        src += 1
        backref_len += 2

        ref = len(output) - ref_offset
        if ref < 0:
            raise ValueError(f"Invalid LZF back-reference offset in {source}")
        for _ in range(backref_len):
            output.append(output[ref])
            ref += 1

    if len(output) != expected_size:
        raise ValueError(
            f"LZF decompressed size mismatch for {source}: "
            f"expected {expected_size}, got {len(output)}"
        )
    return bytes(output)


def _read_binary_compressed_data(
    fh,
    *,
    dtype: np.dtype,
    fields: Sequence[str],
    sizes: Sequence[int],
    counts: Sequence[int],
    points_expected: int,
    source: Path,
) -> np.ndarray:
    size_header = fh.read(8)
    if len(size_header) != 8:
        raise ValueError(f"Missing binary_compressed size header in {source}")
    compressed_size, uncompressed_size = struct.unpack("<II", size_header)
    compressed_payload = fh.read(compressed_size)
    if len(compressed_payload) != compressed_size:
        raise ValueError(
            f"Expected {compressed_size} compressed bytes, read {len(compressed_payload)} from {source}"
        )

    raw = _lzf_decompress(compressed_payload, uncompressed_size, source)
    expected_uncompressed_size = dtype.itemsize * points_expected
    if uncompressed_size != expected_uncompressed_size:
        raise ValueError(
            f"Unexpected uncompressed size for {source}: "
            f"header={uncompressed_size}, expected={expected_uncompressed_size}"
        )

    data = np.empty(points_expected, dtype=dtype)
    offset = 0
    for name, size, count in zip(fields, sizes, counts):
        base_dtype = data.dtype.fields[name][0].base
        if count == 1:
            field_size = size * points_expected
            chunk = raw[offset : offset + field_size]
            data[name] = np.frombuffer(chunk, dtype=base_dtype, count=points_expected)
            offset += field_size
            continue

        values = np.empty((points_expected, count), dtype=base_dtype)
        for idx in range(count):
            field_size = size * points_expected
            chunk = raw[offset : offset + field_size]
            values[:, idx] = np.frombuffer(chunk, dtype=base_dtype, count=points_expected)
            offset += field_size
        data[name] = values

    if offset != len(raw):
        raise ValueError(f"Unused binary_compressed payload bytes in {source}: {len(raw) - offset}")
    return data


def read_pcd(path: Path, *, _converted: bool = False, _source: Optional[Path] = None) -> PCDCloud:
    source_path = path if _source is None else _source

    with path.open("rb") as fh:
        header_lines: List[str] = []
        data_type = None
        while True:
            line_bytes = fh.readline()
            if not line_bytes:
                raise ValueError(f"Unexpected end of file while reading PCD header: {path}")
            line = line_bytes.decode("utf-8", errors="ignore").strip()
            header_lines.append(line)
            if line.upper().startswith("DATA"):
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Malformed DATA line in PCD header: {line}")
                data_type = parts[1].lower()
                break
        if data_type is None:
            raise ValueError(f"PCD header missing DATA specification: {path}")

        metadata, comments = _parse_header(header_lines)

        fields = tuple(metadata.get("FIELDS", []))
        if not fields:
            raise ValueError(f"PCD file {path} does not define FIELDS.")
        fields = _ensure_unique_field_names(fields, source_path)
        try:
            sizes = tuple(int(v) for v in metadata.get("SIZE", []))
            types = tuple(v.upper() for v in metadata.get("TYPE", []))
        except ValueError as exc:
            raise ValueError(f"Invalid SIZE or TYPE values in PCD header: {path}") from exc
        if not sizes or not types or len(sizes) != len(fields) or len(types) != len(fields):
            raise ValueError(f"SIZE/TYPE definitions do not match FIELDS in {path}")

        count_tokens = metadata.get("COUNT")
        if count_tokens is None or len(count_tokens) != len(fields):
            counts = tuple(1 for _ in fields)
        else:
            try:
                counts = tuple(int(v) for v in count_tokens)
            except ValueError as exc:
                raise ValueError(f"Invalid COUNT values in {path}") from exc

        dtype = _build_dtype(fields, sizes, types, counts)

        version = metadata.get("VERSION", ["0.7"])[0]
        try:
            viewpoint = tuple(float(v) for v in metadata.get("VIEWPOINT", ["0", "0", "0", "1", "0", "0", "0"]))
        except ValueError as exc:
            raise ValueError(f"Invalid VIEWPOINT values in {path}") from exc
        try:
            width = int(metadata.get("WIDTH", ["0"])[0])
        except ValueError as exc:
            raise ValueError(f"Invalid WIDTH in {path}") from exc
        try:
            height = int(metadata.get("HEIGHT", ["1"])[0])
        except ValueError as exc:
            raise ValueError(f"Invalid HEIGHT in {path}") from exc
        try:
            declared_points = int(metadata.get("POINTS", [str(width * max(height, 1))])[0])
        except ValueError as exc:
            raise ValueError(f"Invalid POINTS value in {path}") from exc

        points_expected = declared_points if declared_points > 0 else width * max(height, 1)
        if data_type == "binary":
            count = points_expected if points_expected > 0 else -1
            data = np.fromfile(fh, dtype=dtype, count=count)
            if count == -1:
                points_expected = data.shape[0]
            elif data.shape[0] != count:
                raise ValueError(f"Expected {count} points, read {data.shape[0]} from {path}")
            data = data.copy()
        elif data_type == "binary_compressed":
            if _converted:
                raise RuntimeError(f"Converted PCD still marked binary_compressed: {path}")
            data = _read_binary_compressed_data(
                fh,
                dtype=dtype,
                fields=fields,
                sizes=sizes,
                counts=counts,
                points_expected=points_expected,
                source=source_path,
            )
        elif data_type == "ascii":
            data = np.genfromtxt(fh, dtype=dtype, max_rows=declared_points if declared_points > 0 else None)
            if data.size == 0:
                data = np.zeros(0, dtype=dtype)
            elif data.shape == ():
                data = data.reshape(1)
        else:
            raise ValueError(f"Unsupported PCD DATA format '{data_type}' in {path}")

        if declared_points > 0 and data.shape[0] != declared_points:
            raise ValueError(f"Header POINTS={declared_points} does not match actual data ({data.shape[0]}) in {path}")

    data, viewpoint = _apply_viewpoint_transform(data, fields, viewpoint, source_path)

    cloud = PCDCloud(
        fields=fields,
        sizes=sizes,
        types=types,
        counts=counts,
        data_type=data_type,
        version=version,
        viewpoint=viewpoint,
        comments=comments,
        data=data,
        height=height,
    )

    cloud = _strip_ignored_fields(cloud)
    return _canonicalize_field_order(cloud)


def write_pcd(path: Path, cloud: PCDCloud) -> None:
    num_points = cloud.num_points
    height = 1  # merged clouds are treated as unorganized

    lines = list(cloud.comments) if cloud.comments else ["# .PCD v0.7 - Point Cloud Data file format"]
    lines.append(f"VERSION {cloud.version}")
    lines.append("FIELDS " + " ".join(cloud.fields))
    lines.append("SIZE " + " ".join(str(v) for v in cloud.sizes))
    lines.append("TYPE " + " ".join(cloud.types))
    lines.append("COUNT " + " ".join(str(v) for v in cloud.counts))
    lines.append(f"WIDTH {num_points}")
    lines.append(f"HEIGHT {height}")
    lines.append("VIEWPOINT " + " ".join(f"{v:.6f}" for v in cloud.viewpoint))
    lines.append(f"POINTS {num_points}")
    lines.append(f"DATA {cloud.data_type}")

    with path.open("wb") as fh:
        for line in lines:
            fh.write((line + "\n").encode("utf-8"))
        if cloud.data_type == "binary":
            fh.write(cloud.data.tobytes())
        elif cloud.data_type == "ascii":
            array2d = _structured_to_columns(cloud.data, cloud.fields, cloud.counts)
            fmts = _column_formats(cloud.data, cloud.fields, cloud.counts)
            text_buffer = io.StringIO()
            np.savetxt(text_buffer, array2d, fmt=fmts)
            fh.write(text_buffer.getvalue().encode("utf-8"))
        else:
            raise ValueError(f"Unsupported DATA format '{cloud.data_type}' while writing {path}")


def _align_fields(reference: PCDCloud, other: PCDCloud, source: Path) -> PCDCloud:
    ref_field_to_index = {name: idx for idx, name in enumerate(reference.fields)}
    other_field_to_index = {name: idx for idx, name in enumerate(other.fields)}
    if set(ref_field_to_index) != set(other_field_to_index):
        raise ValueError(
            f"Incompatible field sets when merging {source}: "
            f"{sorted(other_field_to_index)} vs {sorted(ref_field_to_index)}"
        )

    dtype = reference.data.dtype
    reordered_data = np.empty(other.num_points, dtype=dtype)
    new_sizes: List[int] = []
    new_types: List[str] = []
    new_counts: List[int] = []

    for name in reference.fields:
        ref_idx = ref_field_to_index[name]
        other_idx = other_field_to_index[name]
        if (
            reference.sizes[ref_idx] != other.sizes[other_idx]
            or reference.types[ref_idx] != other.types[other_idx]
            or reference.counts[ref_idx] != other.counts[other_idx]
        ):
            raise ValueError(f"Field definition mismatch for '{name}' in {source}")
        reordered_data[name] = other.data[name]
        new_sizes.append(other.sizes[other_idx])
        new_types.append(other.types[other_idx])
        new_counts.append(other.counts[other_idx])

    return PCDCloud(
        fields=reference.fields,
        sizes=tuple(new_sizes),
        types=tuple(new_types),
        counts=tuple(new_counts),
        data_type=other.data_type,
        version=other.version,
        viewpoint=other.viewpoint,
        comments=other.comments,
        data=reordered_data,
        height=other.height,
    )


def _ensure_compatible(reference: PCDCloud, other: PCDCloud, source: Path) -> PCDCloud:
    aligned = other
    if reference.fields != other.fields:
        aligned = _align_fields(reference, other, source)
    if (
        reference.sizes != aligned.sizes
        or reference.types != aligned.types
        or reference.counts != aligned.counts
    ):
        raise ValueError(f"Incompatible SIZE/TYPE/COUNT definitions in {source}")
    if reference.data_type != aligned.data_type:
        raise ValueError(f"DATA type mismatch ({aligned.data_type} vs {reference.data_type}) in {source}")
    return aligned


def merge_point_clouds(input_dir: Path) -> PCDCloud:
    merged: Optional[PCDCloud] = None
    for pcd_path in sorted(input_dir.iterdir()):
        if not pcd_path.is_file() or pcd_path.suffix.lower() != ".pcd":
            continue
        print(f"Loading {pcd_path}")
        cloud = read_pcd(pcd_path)
        if merged is None:
            merged = cloud
        else:
            aligned = _ensure_compatible(merged, cloud, pcd_path)
            merged = merged.with_data(np.concatenate([merged.data, aligned.data]), height=1)
    if merged is None:
        raise RuntimeError("No valid PCD files were found to merge.")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge PCD files in a directory into a single PCD."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        type=Path,
        help="Directory containing PCD files to merge.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=(
            "Output PCD file path. Defaults to parent directory of the input "
            "directory with the name global_map_imu.pcd."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    merged = merge_point_clouds(args.input_dir)
    output_path = args.output if args.output is not None else args.input_dir.parent / "global_map_imu.pcd"
    if merged.num_points == 0:
        raise RuntimeError("Merged cloud contains no points; ensure the directory has readable PCD files.")

    print(f"Merged cloud has {merged.num_points} points. Saving to {output_path}")
    write_pcd(output_path, merged)


if __name__ == "__main__":
    main()
