# Taxonomy

12 faithfulness + 12 factuality leaves.

## Faithfulness (video-grounded)

**L1 — Static Entities & Attributes**

| ID | Leaf |
|---|---|
| 1.1 | Entity Recognition |
| 1.2 | Unique Entity Counting |
| 1.3 | Temporal Attributes Recognition |
| 1.4 | Static Attributes Recognition |
| 1.5 | Scene Text Recognition |

**L2 — Dynamic Actions & Motions**

| ID | Leaf |
|---|---|
| 2.1 | Action Recognition |
| 2.2 | Repetitive Action Counting |
| 2.3 | Motion Attributes Recognition |

**L3 — Spatio-Temporal Relations**

| ID | Leaf |
|---|---|
| 3.1 | Spatial Relation Recognition |
| 3.2 | Temporal Relation Recognition |
| 3.3 | State Transition Detection |
| 3.4 | Temporal Localization |

## Factuality (world-knowledge-grounded)

**Know-WHAT — Domain Knowledge**

| ID | Leaf |
|---|---|
| 1.1 | Cultural Event Recognition |
| 1.2 | Historical Background Identification |
| 1.3 | Geospatial Localization |
| 1.4 | Entertainment-Related Recognition |

**Know-HOW — Procedural Knowledge** (binary correct/incorrect; counterfactual via segment shuffling)

| ID | Leaf |
|---|---|
| 2.1 | Electronic |
| 2.2 | Mechanical |
| 2.3 | Domestic |
| 2.4 | Clinical |

**Know-WHY — Physical Knowledge** (synthetic probes only)

| ID | Leaf |
|---|---|
| 3.1 | Newtonian Mechanics |
| 3.2 | Fluid Mechanics |
| 3.3 | Material Property |
| 3.4 | Spatio-Temporal Continuity |

## Order-sensitive subset

TSS is evaluated only on items whose correctness requires chronological order. Default tags (see `configs/taxonomy.yaml`):

- Faithfulness: 2.1, 3.2, 3.3, 3.4 (parts of 2.3 when direction matters).
- Factuality: all four Know-HOW leaves (2.1–2.4).
