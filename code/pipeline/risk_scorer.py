from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.schemas import UserRisk  # noqa: E402
from utils.csv_loader import DataStore  # noqa: E402
from utils.validators import VALID_RISK_FLAGS  # noqa: E402


class RiskScorer:
    def __init__(self, data_store: DataStore):
        self.data_store = data_store

    def score(self, user_id: str) -> UserRisk:
        history = self.data_store.get_user_history(user_id)

        if history is None:
            return UserRisk(
                user_id=user_id,
                user_found=False,
                risk_level="low",
                risk_flags=[],
                risk_summary="No history found for this user",
            )

        past_claim_count = self._safe_int(history.get("past_claim_count", 0))
        rejected_claim_count = self._safe_int(history.get("rejected_claim", 0))
        last_90_days_count = self._safe_int(
            history.get("last_90_days_claim_count", 0)
        )
        history_flags = str(history.get("history_flags", "") or "")
        history_summary = str(history.get("history_summary", "") or "")
        normalized_flags = history_flags.lower()

        risk_flags = []
        if (
            rejected_claim_count >= 2
            or last_90_days_count >= 3
            or "fraud" in normalized_flags
            or "suspicious" in normalized_flags
            or "high_risk" in normalized_flags
        ):
            risk_flags.append("user_history_risk")

        if (
            "manual" in normalized_flags
            or "review" in normalized_flags
            or rejected_claim_count >= 3
        ):
            risk_flags.append("manual_review_required")

        risk_flags = [
            flag for flag in dict.fromkeys(risk_flags) if flag in VALID_RISK_FLAGS
        ]

        if "manual_review_required" in risk_flags or rejected_claim_count >= 3:
            risk_level = "high"
        elif "user_history_risk" in risk_flags:
            risk_level = "medium"
        else:
            risk_level = "low"

        if risk_level == "low":
            risk_summary = "User history shows no unusual patterns"
        elif risk_level == "medium":
            risk_summary = (
                f"Elevated risk: {last_90_days_count} claims in 90 days, "
                f"{rejected_claim_count} rejected"
            )
        else:
            risk_summary = (
                f"High risk: flagged for manual review. {history_summary[:100]}"
            )

        return UserRisk(
            user_id=user_id,
            past_claim_count=past_claim_count,
            rejected_claim_count=rejected_claim_count,
            last_90_days_count=last_90_days_count,
            history_flags=history_flags,
            risk_level=risk_level,
            risk_flags=risk_flags,
            risk_summary=risk_summary,
            user_found=True,
        )

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


if __name__ == "__main__":
    data_store = DataStore()
    scorer = RiskScorer(data_store)

    user_ids = data_store._user_history_df["user_id"].astype(str).head(3).tolist()
    for user_id in user_ids:
        risk = scorer.score(user_id)
        print(f"{user_id}: {risk.risk_level} {risk.risk_flags}")

    missing_user = scorer.score("nonexistent_user_id")
    print(f"{missing_user.user_id}: {missing_user.risk_level} {missing_user.risk_flags}")

    assert missing_user.user_found is False
    print("RISK SCORER OK")
