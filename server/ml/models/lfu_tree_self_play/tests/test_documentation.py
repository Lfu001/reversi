import json
import re
import runpy
from pathlib import Path
from shutil import copytree

import torch

from lfu_tree_self_play.roots import RootRegistry


def test_collection_batch_and_recovery_instructions(tmp_path, monkeypatch):
    package = Path(__file__).resolve().parents[1]
    copytree(package / "configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    namespaces = {}
    for name in ("collection.md", "batch-collection.md", "recovery.md"):
        document = package / "docs" / "guides" / name
        blocks = re.findall(
            r"^```python\n(.*?)^```$",
            document.read_text(encoding="utf-8"),
            flags=re.MULTILINE | re.DOTALL,
        )
        assert blocks, f"no executable instructions in {document}"
        example = tmp_path / Path(name).with_suffix(".py")
        example.write_text("\n".join(blocks), encoding="utf-8")
        with torch.random.fork_rng(devices=[]):
            namespaces[name] = runpy.run_path(str(example))

    collected = namespaces["collection.md"]
    recovered = namespaces["recovery.md"]
    runtime = recovered["runtime"]
    assert runtime.manifest == collected["snapshot"]
    assert not runtime.manifest.progress.consumed_unit_ids
    assert runtime.manifest.inputs.selected == (collected["identity"],)
    assert recovered["loaded"].record == collected["tree"].record
    assert (
        RootRegistry.load("artifacts/roots.json", collected["config"])
        == collected["registry"]
    )
    assert all(
        torch.equal(value, runtime.model.state_dict()[name])
        for name, value in collected["model"].state_dict().items()
    )
    assert runtime.optimizer.state_dict() == collected["optimizer"].state_dict()
    for purpose in (
        "branch_position",
        "action_sampling",
        "sibling_rollout",
        "evaluation",
    ):
        assert (
            getattr(runtime.rng, purpose).getstate()
            == getattr(collected["streams"], purpose).getstate()
        )

    batched = namespaces["batch-collection.md"]
    metrics = json.loads(Path("artifacts/batch-metrics.json").read_text())
    assert metrics["stop_reason"] == "complete"
    assert metrics["completed_units"] == 3
    assert metrics["discarded_units"] == 0
    assert metrics["inference_batches"] > 0
    for unit_id in ("batched-game-0", "batched-game-1", "batched-tree"):
        unit = batched["batch_store"].load(unit_id)
        assert unit.identity.unit_id == unit_id
