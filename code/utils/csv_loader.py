from pathlib import Path
import re

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"


class DataStore:
    def __init__(self):
        self.claims: pd.DataFrame = pd.read_csv(CLAIMS_CSV)
        self.sample_claims: pd.DataFrame = pd.read_csv(SAMPLE_CLAIMS_CSV)
        self._user_history_df: pd.DataFrame = pd.read_csv(USER_HISTORY_CSV)
        self._evidence_requirements_df: pd.DataFrame = pd.read_csv(
            EVIDENCE_REQUIREMENTS_CSV
        )

    def get_user_history(self, user_id: str) -> dict | None:
        lookup_user_id = str(user_id).strip()
        user_ids = self._user_history_df["user_id"].astype(str).str.strip()
        matches = self._user_history_df[user_ids == lookup_user_id]

        if matches.empty:
            return None

        return matches.iloc[0].to_dict()

    def get_evidence_requirements(
        self, claim_object: str, issue_family: str
    ) -> list[str]:
        normalized_object = str(claim_object).strip().lower()
        normalized_issue = str(issue_family).strip().lower()
        if not normalized_issue:
            return []

        requirements = self._evidence_requirements_df
        object_matches = (
            requirements["claim_object"].astype(str).str.strip().str.lower()
        ).isin({normalized_object, "all"})
        issue_terms = {
            normalized_issue,
            normalized_issue.replace("_", " "),
            *normalized_issue.split("_"),
        }
        issue_terms.discard("")
        issue_terms.discard("part")
        issue_terms.discard("damage")
        issue_patterns = [
            re.compile(rf"\b{re.escape(term)}\b") for term in sorted(issue_terms)
        ]
        issue_matches = requirements["applies_to"].astype(str).str.lower().map(
            lambda applies_to: any(
                pattern.search(applies_to) for pattern in issue_patterns
            )
        )
        matches = requirements[object_matches & issue_matches]

        return matches["minimum_image_evidence"].astype(str).tolist()

    def get_all_requirements(self) -> pd.DataFrame:
        return self._evidence_requirements_df


if __name__ == "__main__":
    store = DataStore()
    print(f"claims shape: {store.claims.shape}")
    print(f"sample_claims shape: {store.sample_claims.shape}")
    print(f"user_history shape: {store._user_history_df.shape}")
    print(f"evidence_requirements shape: {store._evidence_requirements_df.shape}")

    first_user_id = store.claims.iloc[0]["user_id"]
    print(f"user history for {first_user_id}: {store.get_user_history(first_user_id)}")
    print(
        "evidence requirements for car/dent: "
        f"{store.get_evidence_requirements('car', 'dent')}"
    )
