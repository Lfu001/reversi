"""Validated conditional terminal returns and conserved path weights."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum
from types import MappingProxyType

from .game import Game
from .record_validation import validate_record
from .records import CollectionRecord


@dataclass(frozen=True)
class ReturnTargets:
    """Immutable ID-keyed results, detached from the record's game states."""

    node_black_returns: Mapping[str, float]
    edge_black_returns: Mapping[str, float]
    node_weights: Mapping[str, float]
    edge_weights: Mapping[str, float]
    node_player_targets: Mapping[str, float]
    edge_parent_targets: Mapping[str, float]

    def __post_init__(self) -> None:
        # Copy even caller-supplied dictionaries before exposing read-only views.
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def aggregate_returns(record: CollectionRecord, game: Game) -> ReturnTargets:
    """Validate a complete unit, average child samples, and propagate weights.

    Each edge's return is its child's black return. Its target uses the actual
    parent turn. Node targets use that node's turn; weights start at one and
    split equally over each parent's actual outgoing samples.
    """
    validate_record(record, game)
    nodes = {node.node_id: node for node in record.nodes}
    outgoing = {node_id: [] for node_id in nodes}
    for edge in record.edges:
        outgoing[edge.parent_node_id].append(edge)

    order = []
    pending = [record.root_node_id]
    node_weights = {record.root_node_id: 1.0}
    edge_weights = {}
    while pending:
        node_id = pending.pop()
        order.append(node_id)
        children = outgoing[node_id]
        if children:
            weight = node_weights[node_id] / len(children)
            for edge in children:
                edge_weights[edge.edge_id] = weight
                node_weights[edge.child_node_id] = weight
                pending.append(edge.child_node_id)

    node_returns = {}
    for node_id in reversed(order):
        children = outgoing[node_id]
        if children:
            node_returns[node_id] = fsum(
                node_returns[edge.child_node_id] for edge in children
            ) / len(children)
        else:
            node_returns[node_id] = float(nodes[node_id].black_result)

    edge_returns = {
        edge.edge_id: node_returns[edge.child_node_id] for edge in record.edges
    }
    return ReturnTargets(
        node_black_returns=node_returns,
        edge_black_returns=edge_returns,
        node_weights=node_weights,
        edge_weights=edge_weights,
        node_player_targets={
            node_id: value * nodes[node_id].state.to_play
            for node_id, value in node_returns.items()
        },
        edge_parent_targets={
            edge.edge_id: edge_returns[edge.edge_id]
            * nodes[edge.parent_node_id].state.to_play
            for edge in record.edges
        },
    )
