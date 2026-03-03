import json

from config import get_config
from detectors.interfaces.classification import DecisionResult, DecisionState

# Hardcoded policy version mapping to this code implementation
POLICY_VERSION = "v1"

# Reason Codes
REASON_LOW_BBOX_QUALITY = "LOW_BBOX_QUALITY"
REASON_LOW_SPECIES_CONF = "LOW_SPECIES_CONF"
REASON_NO_SPECIES_CONF = "NO_SPECIES_CONF"
REASON_HIGH_UNKNOWN_SCORE = "HIGH_UNKNOWN_SCORE"


class DecisionPolicyService:
    """
    Central, deterministic rule engine for BBox/Species/Unknown decisions.
    Evaluates raw signals against configured thresholds to produce a DecisionResult.
    """

    def __init__(self, config: dict | None = None):
        self._config = config or get_config()
        self.bbox_threshold = float(self._config.get("BBOX_QUALITY_THRESHOLD", 0.40))
        self.species_threshold = float(self._config.get("SPECIES_CONF_THRESHOLD", 0.70))
        self.unknown_threshold = float(
            self._config.get("UNKNOWN_SCORE_THRESHOLD", 0.60)
        )

    def evaluate(
        self,
        bbox_quality: float | None,
        species_conf: float | None,
        unknown_score: float | None,
    ) -> DecisionResult:
        """
        Evaluates the combination of signals into a final decision state.

        Args:
            bbox_quality: Float [0, 1] representing bounding box heuristic quality.
            species_conf: Float [0, 1] representing the top-1 species classification confidence.
            unknown_score: Float [0, 1] representing out-of-distribution/uncertainty.

        Returns:
            DecisionResult holding the final DecisionState and a list of reason codes.
        """
        reasons = []

        is_bad_bbox = bbox_quality is not None and bbox_quality < self.bbox_threshold
        is_high_unknown = (
            unknown_score is not None and unknown_score >= self.unknown_threshold
        )
        is_low_species = (
            species_conf is not None and species_conf < self.species_threshold
        )
        # Missing species confidence means classification never ran or failed.
        # This must NEVER be treated as confirmed.
        is_missing_species = species_conf is None

        if is_bad_bbox:
            reasons.append(REASON_LOW_BBOX_QUALITY)
        if is_high_unknown:
            reasons.append(REASON_HIGH_UNKNOWN_SCORE)
        if is_low_species:
            reasons.append(REASON_LOW_SPECIES_CONF)
        if is_missing_species:
            reasons.append(REASON_NO_SPECIES_CONF)

        # 1. Unknown has high priority if signals heavily point to it
        if is_high_unknown:
            state = DecisionState.UNKNOWN

        # 2. Bad bounding box overrides high species confidence
        elif is_bad_bbox:
            # If everything else is also bad, it might just be garbage (rejected/unknown)
            if is_low_species or is_missing_species:
                state = DecisionState.REJECTED
            else:
                # E.g. species conf is high, but bbox is bad - we shouldn't blindly trust it
                state = DecisionState.UNCERTAIN

        # 3. If BBox is ok, but species confidence is too low or missing
        elif is_low_species or is_missing_species:
            state = DecisionState.UNCERTAIN

        # 4. Happy Path — only when all signals are present and above threshold
        else:
            state = DecisionState.CONFIRMED

        return DecisionResult(
            decision_state=state,
            bbox_quality=bbox_quality,
            species_conf=species_conf,
            unknown_score=unknown_score,
            reasons_json=json.dumps(reasons) if reasons else "[]",
            policy_version=POLICY_VERSION,
        )
