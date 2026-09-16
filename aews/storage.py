import json


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(fs, path):
    return json.loads(fs.head(path, 16 * 1024 * 1024))


def write_once_json(fs, path, value):
    if fs.exists(path):
        if read_json(fs, path) != value:
            raise RuntimeError(f"Unknown or conflicting existing manifest: {path}")
        return
    parent = path.rsplit("/", 1)[0]
    if not fs.mkdirs(parent):
        raise RuntimeError(f"Could not create manifest parent: {parent}")
    if not fs.put(path, canonical_json(value), overwrite=False):
        raise RuntimeError(f"Could not persist manifest: {path}")
    if read_json(fs, path) != value:
        raise RuntimeError(f"Manifest read-back mismatch: {path}")


def schema_signature(frame):
    return canonical_json(frame.schema.jsonValue())


def require_equivalent(left, right, description):
    if schema_signature(left) != schema_signature(right):
        raise RuntimeError(f"Schema mismatch: {description}")
    left_count, right_count = left.count(), right.count()
    if left_count != right_count:
        raise RuntimeError(f"Row count mismatch: {description}: {left_count} != {right_count}")
    if left.exceptAll(right).limit(1).count() or right.exceptAll(left).limit(1).count():
        raise RuntimeError(f"Row content mismatch: {description}")
    return left_count


def archive_delta(spark, fs, frame, path):
    if not fs.exists(path):
        frame.write.format("delta").mode("errorifexists").save(path)
    archived = spark.read.format("delta").load(path)
    require_equivalent(frame, archived, path)
    return archived


