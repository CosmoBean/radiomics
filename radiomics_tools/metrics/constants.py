"""Shared constants for compartment-based report metrics."""

LABEL_MAP = {
    "et": 1,
    "netc": 2,
    "snhf": 3,
    "rc": 4,
}
WHOLE_TUMOR_LABELS = tuple(LABEL_MAP.values())
PREDICTIVE_TUMOR_LABELS = (
    LABEL_MAP["et"],
    LABEL_MAP["netc"],
    LABEL_MAP["snhf"],
)
EPSILON = 1e-8

