# Atomic complete-unit dataset store (T08)

`DatasetStore` persists independent completed games and trees whose every leaf
is terminal. It never connects data to the trainer; G1 remains a separate gate.

```python
store = DatasetStore(root, experiment_id=experiment_id(config), game=game)
identity = store.publish(record, aggregate_returns(record, game))
unit = store.load(record.unit_id, model_generation=record.model_generation)
identities = store.enumerate_units(model_generation=record.model_generation)
quarantined_paths = store.quarantine_incomplete()
```

Import the store from `lfu_tree_self_play.dataset_store`, and the explicit
`encode_unit(record, targets, game) -> bytes` / `decode_unit(data, game) -> DatasetUnit`
codec from `lfu_tree_self_play.dataset_codec`. `DatasetUnit` contains `record`,
`targets`, and an immutable `UnitIdentity(unit_id, experiment_id, model_generation,
checksum)`. T09 should pin these identities in its own input manifest; this store
has no separate consumption position or mutable identity index.

Pass the existing T02 `experiment_id(config)` without inventing another settings
hash. The first publication atomically pins `experiment.json` under the writer
lock; every later append rechecks this immutable root binding before writing.
Two first writers with different configurations cannot both append. Existing
units in a root without a binding are validated before the binding is created.
The binding is configuration metadata, not another identity or consumption index.
Publish, load, enumeration, and quarantine refuse another bound experiment ID. Generation
is immutable record metadata: `load(..., model_generation=N)` rejects a mismatch;
enumeration with this argument filters verified historical units to generation N.
Every published entry is validated before filtering so corruption fails closed.
Enumeration returns a lexical ID ordered snapshot; a concurrent publication may
appear on the next call. Caller manifests must pin the returned identities rather
than assume directory enumeration is an input transaction.

## Wire and integrity contract

Each `published/<unit_id>.json` file is one version-1 envelope with exactly
`payload` and `checksum`. Payload has exactly `version`, `record`, and `targets`.
The checksum is lowercase SHA256 over `json.dumps(payload, sort_keys=True,
separators=(",", ":"), allow_nan=False).encode()` (UTF-8 and default `ensure_ascii=True`).
The enclosing checksum is excluded from these bytes. Arrays retain record order;
target dictionary keys are sorted by the JSON canonicalization.

Game states use explicit four-plane `(4, 8, 8)` numeric arrays, losslessly restored
as float32 `GameState` instances. Tuple record fields are explicitly rebuilt.
Unknown fields and versions, duplicate JSON keys, booleans in numeric fields,
nonfinite values, invalid game transitions, incomplete leaves, mixed generations,
and targets unequal to `aggregate_returns` are rejected. Every target mapping
must be a JSON object, and every tuple field must be a JSON array before explicit
reconstruction; arrays of key/value pairs and arbitrary iterable coercions reject. Checksums detect damage;
semantic validation still applies when a damaged payload has a recalculated checksum.
This is corruption detection, not authentication against an attacker rewriting files.

Encoding copies all six supplied target mappings into ordinary dictionaries,
requires exact Python string keys and finite Python int/float values excluding
booleans, and compares each dictionary with freshly aggregated validated returns.
It first snapshots the base record schema and rebuilds exact `GameState` instances
from copied numeric observations, rejecting invalid planes or lossy float32
conversion. Aggregation and serialization use that same snapshot, so custom
record serializers or state properties cannot change persisted game semantics.
Publication obtains bytes and identity from that same validated payload, without
decoding its own new bytes. Loading, enumeration, and replay of an existing file
still perform full checksum, record, game, and target validation.

Unit IDs must match `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}`. Repeating an ID with the same
canonical payload returns the original identity. A different payload is a replay
conflict and never overwrites the original, including changes to round metadata.
Corrupt existing data causes a failure instead of replacement.

## Publication and recovery

Cooperating writers use `fcntl.flock` on `writer.lock`. Validation precedes staging;
the writer creates a unique staging file, writes/flushed/fsyncs it, fsyncs staging,
then atomically renames to published and fsyncs both directories. Identical replay fsyncs published and staging directories before success,
finishing durability work interrupted after rename. Creating a new root also
syncs each newly created directory entry through its parent. Readers use only
published files and never acquire the writer lock. There is no multi-file unit
whose metadata could become visible before its data.

A process stopped before rename leaves an invisible staging file. Explicit
`quarantine_incomplete()` holds the writer lock, moves abandoned staging entries
to unique quarantine paths, and syncs the directories. It never promotes them;
replaying collection may publish the same ID. A process stopped after rename
leaves a complete visible unit and replay succeeds. Quarantine never silently
moves published corruption or incompatible configuration out of the dataset.

The guarantee applies to cooperating POSIX writers on one local filesystem with
working flock, atomic rename, and fsync. Network filesystems, hostile external file
mutation, and physically failing storage are outside this contract. A successful
publish has completed fsyncs; process-stop tests do not establish hardware power-loss
behavior. There is no policy deciding whether historical generations are suitable
for a particular training update.

## Verification

`PYTHONPATH=src python -m pytest tests/test_dataset_store.py -q` covers exact
round-trip (uneven trees, duplicate samples, H=0/H=1 and independent forced passes),
checksum and resealed semantic corruption, replay/conflict, configuration/generation
rejection, unsafe IDs, abandoned-stage quarantine, and subprocess interruption
immediately before/after rename. During each paused child writer, the parent repeatedly
reads while the writer still holds flock; it sees absence or a fully validated unit.
