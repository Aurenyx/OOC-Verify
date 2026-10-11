import unittest
from app.services.decision_service import make_final_decision


class TestDecisionFusionSafety(unittest.TestCase):
    """
    Unit tests ensuring the OOC-Verify decision fusion engine safely handles
    missing, failed, conflicting, or unavailable upstream signals without
    silently defaulting to Genuine.
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

    def test_conflicting_qwen_and_alignment_returns_inconclusive(self):
        """
        When Qwen says Genuine but Alignment classifier says Misleading
        (with confidence < 0.90 and no strong external evidence),
        the conflicting visual signals must produce Inconclusive instead of Genuine.
        """
        decision = make_final_decision(
            alignment_prediction="Misleading",
            alignment_confidence=0.68,
            qwen_response="Prediction: Genuine\nReason: Image appears plausible for caption.",
            evidence_items=[],
            caption="A sports car parked in front of a modern house",
        )
        self.assertEqual(decision["prediction"], "Inconclusive")
        self.assertEqual(decision["confidence_score"], 0.50)
        self.assertIn("alignment classifier indicated a potentially misleading mismatch", decision["reason"])

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
        """When retrieved evidence strongly contradicts the claim, return Misleading."""
        contradicting_evidence = [
            {"relation": "contradicts", "relevance": 0.85, "source": "AP News", "title": "Fact check debunk"}
        ]
        decision = make_final_decision(
            alignment_prediction="Genuine",
            alignment_confidence=0.80,
            qwen_response="Prediction: Genuine\nReason: Visually looks like an explosion.",
            evidence_items=contradicting_evidence,
            caption="Recent explosion at city refinery",
        )
        self.assertEqual(decision["prediction"], "Misleading")
        self.assertIn("Retrieved evidence strongly contradicts", decision["reason"])

    def test_preserves_legitimate_misleading_from_qwen(self):
        """When Qwen identifies visual inconsistency, return Misleading."""
        decision = make_final_decision(
            alignment_prediction="Genuine",
            alignment_confidence=0.55,
            qwen_response="Prediction: Misleading\nReason: Caption claims nighttime snow, but image is sunny beach.",
            evidence_items=[],
            caption="Heavy snowstorm in Alaska last night",
        )
        self.assertEqual(decision["prediction"], "Misleading")

    def test_preserves_legitimate_ooc_protection(self):
        """
        When external evidence confirms event, but caption is contextual
        and alignment classifier flags a mismatch, return Misleading (out of context).
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
        self.assertEqual(decision["prediction"], "Misleading")
        self.assertIn("submitted image may be out of context", decision["reason"])

    def test_preserves_qwen_unavailable_with_alignment_misleading(self):
        """When Qwen is unavailable but Alignment is Misleading, return Misleading."""
        decision = make_final_decision(
            alignment_prediction="Misleading",
            alignment_confidence=0.75,
            qwen_response="Prediction: Pending\nReason: Timeout",
            evidence_items=[],
            caption="Breaking news claim",
        )
        self.assertEqual(decision["prediction"], "Misleading")

    def test_rule_7c_qwen_genuine_alignment_unavailable_without_evidence_returns_inconclusive(self):
        """
        Rule 7c: When Qwen predicts Genuine but the alignment classifier is unavailable
        and external evidence is insufficient, return Inconclusive rather than Genuine.
        """
        decision = make_final_decision(
            alignment_prediction="Pending",
            alignment_confidence=0.0,
            qwen_response="Prediction: Genuine\nReason: Image appears visually consistent.",
            evidence_items=[],
            caption="A generic scene described in text",
        )
        self.assertEqual(decision["prediction"], "Inconclusive")
        self.assertEqual(decision["confidence_score"], 0.50)
        self.assertIn("alignment classifier was unavailable and external evidence is insufficient", decision["reason"])

    def test_rule_4_qwen_genuine_alignment_unavailable_with_strong_evidence_preserves_genuine(self):
        """
        Rule 4: When Qwen predicts Genuine and alignment is unavailable,
        but strong external evidence supports the caption, preserve Genuine.
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


if __name__ == "__main__":
    unittest.main()
