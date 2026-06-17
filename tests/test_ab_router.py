"""
Unit tests for the A/B testing router — variant assignment and stats.
"""

import pytest

from src.ab_testing.router import ABRouter


class TestVariantAssignment:
    """Variant assignment must be deterministic and respect traffic split."""

    def test_same_entity_same_variant(self):
        """Same entity_id always gets the same variant within an experiment."""
        v1 = ABRouter._assign_variant("user_abc", "exp_1", 0.9)
        v2 = ABRouter._assign_variant("user_abc", "exp_1", 0.9)
        assert v1 == v2

    def test_variant_in_valid_set(self):
        variant = ABRouter._assign_variant("user_xyz", "exp_1", 0.9)
        assert variant in ("champion", "challenger")

    def test_traffic_split_approximation(self):
        """With 90% champion traffic and 10k users, ~90% should hit champion."""
        champion_count = sum(
            1 for i in range(10_000)
            if ABRouter._assign_variant(f"user_{i}", "traffic_test", 0.9) == "champion"
        )
        # Allow ±3% tolerance
        assert 0.87 <= champion_count / 10_000 <= 0.93

    def test_different_experiments_independent(self):
        """Same entity_id can get different variants in different experiments."""
        variants = {
            ABRouter._assign_variant("user_fixed", f"exp_{i}", 0.5)
            for i in range(20)
        }
        # Not all variants should be the same — different experiments should differ
        assert len(variants) == 2   # at least one champion and one challenger seen


class TestStatisticalSignificance:
    def test_insufficient_data_returns_none(self):
        champ = {"tp": 5, "fp": 3, "fn": 2, "tn": 10}
        chall = {"tp": 4, "fp": 4, "fn": 3, "tn": 9}
        result = ABRouter._significance_test(champ, chall)
        assert result is None   # n < 30

    def test_identical_groups_not_significant(self):
        champ = {"tp": 500, "fp": 50, "fn": 100, "tn": 850}
        chall = {"tp": 500, "fp": 50, "fn": 100, "tn": 850}
        p = ABRouter._significance_test(champ, chall)
        assert p is not None
        assert p > 0.05   # no difference → not significant

    def test_clear_winner_is_significant(self):
        champ = {"tp": 400, "fp": 200, "fn": 100, "tn": 300}   # precision 0.67
        chall = {"tp": 500, "fp": 50,  "fn": 100, "tn": 350}   # precision 0.91
        p = ABRouter._significance_test(champ, chall)
        assert p is not None
        assert p < 0.05   # large difference → significant

    def test_recommend_promote_challenger(self):
        champ = {"precision": 0.70}
        chall = {"precision": 0.91}
        rec = ABRouter._recommend(champ, chall, p_value=0.02)
        assert rec == "promote_challenger"

    def test_recommend_keep_champion_not_significant(self):
        champ = {"precision": 0.70}
        chall = {"precision": 0.72}
        rec = ABRouter._recommend(champ, chall, p_value=0.18)
        assert rec == "keep_champion"
