import unittest
from app.services.decision_service import make_final_decision


class TestDecisionFusionSafety(unittest.TestCase):
    """
    Unit tests ensuring the OOC-Verify decision fusion engine implements
    the updated prediction policy:
    1. Valid Qwen prediction (Genuine or Misleading) determines the final prediction.
    2. Alignment disagreement or missing evidence does not override valid Qwen predictions.
    3. Inconclusive is used safely as fallback when Qwen fails or output is invalid/inconclusive.
    """

    def test_missing_qwen_output_returns_inconclusive(self):
        """When Qwen fails or returns Unknown/Pending, do not default to Genuine."""
        qwen_failed_response = (
            "Prediction: Unknown\n"
            "Reason: MLLM analysis failed: CUDA out of memory\n"
            "Visual Evidence: Unavailable"
        )
        decision = make_final_decision(
            alignment_prediction="Genuine",
            alignment_confidence=0.55,
            qwen_response=qwen_failed_response,
            evidence_items=[],
            caption="A crowd gathers at a protest in the city square",
        )
        self.assertEqual(decision["prediction"], "Inconclusive")
        self.assertIn("Insufficient evidence or unavailable multimodal signals", decision["reason"])

    def test_completely_unavailable_signals_returns_inconclusive(self):
        """When Qwen, Alignment, and Evidence are all unavailable/empty, return Inconclusive."""
        decision = make_final_decision(
            alignment_prediction="Pending",
            alignment_confidence=0.0,
            qwen_response="Prediction: Pending\nReason: Not run",
            evidence_items=[],
            caption="A generic image without context",
        )
        self.assertEqual(decision["prediction"], "Inconclusive")
        self.assertEqual(decision["qwen_prediction"], "Pending")

    def test_failed_evidence_with_missing_qwen_returns_inconclusive(self):
        """When evidence retrieval fails/empty and Qwen is unparsed, return Inconclusive."""
        decision = make_final_decision(
            alignment_prediction="Genuine",
            alignment_confidence=0.60,
            qwen_response="Unparseable gibberish output",
            evidence_items=[],
            caption="Breaking news event reported yesterday",
        )
        self.assertEqual(decision["prediction"], "Inconclusive")

    def test_qwen_genuine_with_alignment_misleading_returns_genuine(self):
        """
        When Qwen says Genuine but Alignment classifier says Misleading,
        the valid Qwen prediction determines the final prediction (Genuine).
        """
        decision = make_final_decision(
            alignment_prediction="Misleading",
            alignment_confidence=0.68,
            qwen_response="Prediction: Genuine\nReason: Image appears plausible for caption.",
            evidence_items=[],
            caption="A sports car parked in front of a modern house",
        )
        self.assertEqual(decision["prediction"], "Genuine")
        self.assertEqual(decision["qwen_prediction"], "Genuine")
        self.assertEqual(decision["confidence_score"], 0.60)
        self.assertIn("visual reasoning confirmed genuine consistency", decision["reason"])

    def test_preserves_legitimate_genuine_agreement(self):
        """When Qwen and Alignment both agree on Genuine, return Genuine."""
        decision = make_final_decision(
            alignment_prediction="Genuine",
            alignment_confidence=0.80,
            qwen_response="Prediction: Genuine\nReason: Visual objects match description.",
            evidence_items=[],
            caption="Two dogs playing with a ball in a green park",
        )
        self.assertEqual(decision["prediction"], "Genuine")
        self.assertGreater(decision["confidence_score"], 0.70)

    def test_preserves_legitimate_genuine_with_strong_evidence(self):
        """When strong external evidence supports the caption and Qwen agrees, return Genuine."""
        strong_evidence = [
            {"relation": "supports", "relevance": 0.90, "source": "Reuters", "title": "Official release"}
        ]
        decision = make_final_decision(
            alignment_prediction="Genuine",
            alignment_confidence=0.75,
            qwen_response="Prediction: Genuine\nReason: Consistent",
            evidence_items=strong_evidence,
            caption="President addresses summit in Geneva",
        )
        self.assertEqual(decision["prediction"], "Genuine")
        self.assertGreaterEqual(decision["support_score"], 0.70)

    def test_preserves_legitimate_misleading_from_evidence_contradiction(self):
        """When Qwen identifies Misleading and evidence contradicts the claim, return Misleading."""
        contradicting_evidence = [
            {"relation": "contradicts", "relevance": 0.85, "source": "AP News", "title": "Fact check debunk"}
        ]
        decision = make_final_decision(
            alignment_prediction="Misleading",
            alignment_confidence=0.80,
            qwen_response="Prediction: Misleading\nReason: Visually does not match explosion claim.",
            evidence_items=contradicting_evidence,
            caption="Recent explosion at city refinery",
        )
        self.assertEqual(decision["prediction"], "Misleading")
        self.assertGreaterEqual(decision["confidence_score"], 0.75)

    def test_preserves_legitimate_misleading_from_qwen(self):
        """When Qwen identifies visual inconsistency, return Misleading despite alignment Genuine."""
        decision = make_final_decision(
            alignment_prediction="Genuine",
            alignment_confidence=0.55,
            qwen_response="Prediction: Misleading\nReason: Caption claims nighttime snow, but image is sunny beach.",
            evidence_items=[],
            caption="Heavy snowstorm in Alaska last night",
        )
        self.assertEqual(decision["prediction"], "Misleading")
        self.assertEqual(decision["confidence_score"], 0.60)

    def test_preserves_qwen_genuine_with_contextual_claim(self):
        """
        When Qwen predicts Genuine for a contextual claim, alignment mismatch does not override.
        """
        supporting_evidence = [
            {"relation": "supports", "relevance": 0.80, "source": "BBC", "title": "G7 leaders meet in Berlin"}
        ]
        decision = make_final_decision(
            alignment_prediction="Misleading",
            alignment_confidence=0.70,
            qwen_response="Prediction: Genuine\nReason: People sitting at a conference table.",
            evidence_items=supporting_evidence,
            caption="G7 leaders summit meeting in berlin 2026",
        )
        self.assertEqual(decision["prediction"], "Genuine")

    def test_preserves_qwen_unavailable_with_alignment_misleading(self):
        """When Qwen is unavailable, alignment alone cannot decide; return Inconclusive."""
        decision = make_final_decision(
            alignment_prediction="Misleading",
            alignment_confidence=0.75,
            qwen_response="Prediction: Pending\nReason: Timeout",
            evidence_items=[],
            caption="Breaking news claim",
        )
        self.assertEqual(decision["prediction"], "Inconclusive")
        self.assertEqual(decision["confidence_score"], 0.50)

    def test_qwen_genuine_alignment_unavailable_without_evidence_preserves_genuine(self):
        """
        When Qwen predicts Genuine and alignment classifier is unavailable,
        preserve Genuine (base confidence 0.60).
        """
        decision = make_final_decision(
            alignment_prediction="Pending",
            alignment_confidence=0.0,
            qwen_response="Prediction: Genuine\nReason: Image appears visually consistent.",
            evidence_items=[],
            caption="A generic scene described in text",
        )
        self.assertEqual(decision["prediction"], "Genuine")
        self.assertEqual(decision["confidence_score"], 0.60)

    def test_rule_4_qwen_genuine_alignment_unavailable_with_strong_evidence_preserves_genuine(self):
        """
        When Qwen predicts Genuine and alignment is unavailable,
        and strong external evidence supports the caption, preserve Genuine with evidence boost.
        """
        strong_evidence = [
            {"relation": "supports", "relevance": 0.85, "source": "BBC News", "title": "Verified event"}
        ]
        decision = make_final_decision(
            alignment_prediction="Pending",
            alignment_confidence=0.0,
            qwen_response="Prediction: Genuine\nReason: Visual consistency confirmed.",
            evidence_items=strong_evidence,
            caption="Verified world news event",
        )
        self.assertEqual(decision["prediction"], "Genuine")
        self.assertGreaterEqual(decision["support_score"], 0.70)
        self.assertGreater(decision["confidence_score"], 0.70)

    def test_flooded_road_compatible_caption_is_genuine(self):
        """
        User problem case: Qwen Genuine + alignment Misleading (0.5506) -> Genuine.
        """
        decision = make_final_decision(
            alignment_prediction="Misleading",
            alignment_confidence=0.5506,
            qwen_response=(
                "Prediction: Genuine\n"
                "Reason: The image shows a flooded road."
            ),
            evidence_items=[],
            caption="Residents stand beside a road flooded after heavy rain.",
        )

        self.assertEqual(decision["prediction"], "Genuine")
        self.assertEqual(decision["confidence_score"], 0.60)

    def test_flooded_road_vs_snowy_temple_is_misleading(self):
        decision = make_final_decision(
            alignment_prediction="Misleading",
            alignment_confidence=0.9729,
            qwen_response=(
                "Prediction: Misleading\n"
                "Reason: The image shows a flooded road, not a snowy temple."
            ),
            evidence_items=[],
            caption=(
                "A group of tourists gather outside a snow-covered "
                "mountain temple during a winter festival."
            ),
        )

        self.assertEqual(decision["prediction"], "Misleading")
        self.assertGreater(decision["confidence_score"], 0.70)

    def test_unverified_mumbai_contextual_claim_is_misleading(self):
        """
        When Qwen predicts Misleading, contextual claim does not demote to Inconclusive.
        """
        decision = make_final_decision(
            alignment_prediction="Misleading",
            alignment_confidence=0.5000,
            qwen_response=(
                "Prediction: Misleading\n"
                "Reason: The location and date cannot be established."
            ),
            evidence_items=[
                {"relation": "insufficient", "relevance": 0.3864}
            ],
            caption=(
                "Residents gather beside a flooded road in Mumbai "
                "after heavy monsoon rainfall in July 2025."
            ),
        )

        self.assertEqual(decision["prediction"], "Misleading")
        self.assertEqual(decision["qwen_prediction"], "Misleading")


if __name__ == "__main__":
    unittest.main()
