"""Unit and end-to-end tests.  Run:  python -m unittest discover -s tests -v"""

import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraud_detection.api import build_server, parse_api_keys  # noqa: E402
from fraud_detection.cases import CaseStore  # noqa: E402
from fraud_detection.data import generate_transactions  # noqa: E402
from fraud_detection.features import CustomerProfileStore, build_features  # noqa: E402
from fraud_detection.modeling import best_f1_threshold, classification_metrics, run_training, time_split  # noqa: E402
from fraud_detection.preprocess import clean_transactions, smote  # noqa: E402
from fraud_detection.rules import RuleConfig, decide  # noqa: E402
from fraud_detection.scoring import Scorer  # noqa: E402

H = 3600.0


class FeatureTests(unittest.TestCase):
    def test_behavioural_features_use_only_past_transactions(self):
        s = CustomerProfileStore()
        f1 = s.compute_and_update("C1", 0 * H, 1000, "Chennai", "D1", "grocery", "pos")
        f2 = s.compute_and_update("C1", 1 * H, 3000, "Chennai", "D1", "grocery", "pos")
        f3 = s.compute_and_update("C1", 2 * H, 2000, "Mumbai", "NEW", "travel", "online")
        self.assertEqual(f1["txn_index"], 0)
        self.assertEqual(f1["amount_ratio"], 1.0)           # no history -> neutral ratio
        self.assertEqual(f1["new_device"], 0)               # first device is not "new"
        self.assertAlmostEqual(f2["amount_ratio"], 3.0)     # 3000 vs prior mean 1000
        self.assertEqual(f2["txn_count_24h"], 1)
        self.assertAlmostEqual(f2["time_since_last_min"], 60.0)
        self.assertEqual((f3["location_change"], f3["new_device"]), (1, 1))
        self.assertAlmostEqual(f3["amount_ratio"], 1.0)     # 2000 vs mean(1000, 3000) = 2000
        self.assertEqual(f3["txn_count_24h"], 2)

    def test_24h_window_expires(self):
        s = CustomerProfileStore()
        s.compute_and_update("C1", 0, 100, "A", "D", "fuel", "pos")
        f = s.compute_and_update("C1", 25 * H, 100, "A", "D", "fuel", "pos")
        self.assertEqual(f["txn_count_24h"], 0)

    def test_batch_features_match_online_replay(self):
        raw = generate_transactions(n_transactions=3000, seed=3)
        clean, _ = clean_transactions(raw)
        batch = build_features(clean)
        store = CustomerProfileStore()
        last = batch.iloc[-1]
        # replay everything except the last row, then compute the last row online
        build_features(clean.sort_values(["timestamp", "transaction_id"]).iloc[:-1], store)
        online = store.compute_and_update(
            last.customer_id, (last.timestamp - pd.Timestamp("1970-01-01")).total_seconds(),
            last.amount, last.city, last.device_id, last.merchant_category, last.channel,
        )
        for k in ("txn_count_24h", "amount_ratio", "location_change", "new_device", "txn_index"):
            self.assertAlmostEqual(float(last[k]), float(online[k]), places=6, msg=k)


class PreprocessTests(unittest.TestCase):
    def test_cleaning_removes_duplicates_and_invalid_rows(self):
        raw = generate_transactions(n_transactions=2000, seed=1)
        clean, report = clean_transactions(raw)
        self.assertTrue(clean["transaction_id"].is_unique)
        self.assertTrue((clean["amount"] > 0).all())
        self.assertFalse(clean["merchant_category"].isna().any())
        self.assertGreater(report["duplicates_removed"], 0)

    def test_smote_balances_training_data_only_with_new_minority_points(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (900, 3)), rng.normal(4, 0.5, (30, 3))])
        y = np.array([0] * 900 + [1] * 30)
        Xr, yr = smote(X, y, sampling_ratio=0.5, random_state=0)
        self.assertEqual(int((yr == 0).sum()), 900)
        self.assertEqual(int((yr == 1).sum()), 450)
        synth = Xr[len(X):]
        self.assertTrue((synth.min(axis=0) >= X[y == 1].min(axis=0) - 1e-9).all())  # inside minority hull box
        self.assertTrue((synth.max(axis=0) <= X[y == 1].max(axis=0) + 1e-9).all())

    def test_time_split_has_no_overlap_or_lookahead(self):
        feat = build_features(clean_transactions(generate_transactions(n_transactions=3000, seed=2))[0])
        tr, va, te = time_split(feat)
        self.assertLessEqual(tr.timestamp.max(), va.timestamp.min())
        self.assertLessEqual(va.timestamp.max(), te.timestamp.min())


class MetricTests(unittest.TestCase):
    def test_metrics_on_known_example(self):
        y = np.array([1, 1, 0, 0, 0, 0])
        p = np.array([0.9, 0.4, 0.8, 0.1, 0.2, 0.3])
        m = classification_metrics(y, p, 0.5)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 1, 1, 3))
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["accuracy"], 4 / 6)
        self.assertTrue(0.0 <= best_f1_threshold(y, p) <= 1.0)


class RulesTests(unittest.TestCase):
    RULES = RuleConfig(hard_block_amount=1000, velocity_review_count=5, blocklist=["BAD"])
    QUIET = {"new_device": 0, "txn_count_24h": 0}

    def decision(self, p, feats=None, customer="C", amount=10):
        return decide(p, 0.4, 0.8, feats or self.QUIET, customer, amount, self.RULES)["decision"]

    def test_levels_and_escalation(self):
        self.assertEqual(self.decision(0.1), "approve")
        self.assertEqual(self.decision(0.5), "review")
        self.assertEqual(self.decision(0.9), "block")
        self.assertEqual(self.decision(0.1, customer="BAD"), "block")
        self.assertEqual(self.decision(0.1, {"new_device": 1, "txn_count_24h": 0}, amount=5000), "block")
        self.assertEqual(self.decision(0.1, {"new_device": 0, "txn_count_24h": 6}), "review")

    def test_rules_never_downgrade_a_model_decision(self):
        self.assertEqual(self.decision(0.95, {"new_device": 0, "txn_count_24h": 6}), "block")


class CaseStoreTests(unittest.TestCase):
    def test_case_lifecycle_and_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CaseStore(f"{tmp}/c.db")
            res = {"transaction_id": "T1", "customer_id": "C1", "fraud_probability": 0.9,
                   "risk_level": "high", "decision": "block", "reasons": ["x"]}
            cid = store.create(res, 123.0)
            self.assertEqual(store.stats()["open"], 1)
            self.assertEqual(store.resolve(cid, "confirmed_fraud", "asha")["status"], "confirmed_fraud")
            self.assertEqual(store.feedback_labels(), [{"transaction_id": "T1", "is_fraud": 1}])
            with self.assertRaises(ValueError):
                store.resolve(cid, "open")
            self.assertIsNone(store.resolve(999, "dismissed"))


class EndToEndAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        run_training(n_transactions=9000, seed=7, fast=True, out_dir=str(cls.root / "models"),
                     reports_dir=str(cls.root / "reports"), data_path=None, verbose=False)
        scorer = Scorer.load(models_root=str(cls.root / "models"))
        cls.server = build_server("127.0.0.1", 0, scorer, db_path=str(cls.root / "c.db"),
                                  outbox_path=str(cls.root / "o.jsonl"),
                                  api_keys=parse_api_keys("k-admin:admin,k-cust:customer"),
                                  reports_dir=str(cls.root / "reports"))
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def call(self, method, path, body=None, key=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["X-API-Key"] = key
        c.request(method, path, json.dumps(body) if body is not None else None, headers)
        r = c.getresponse()
        data = json.loads(r.read() or b"{}")
        c.close()
        return r.status, data

    def test_health_needs_no_key(self):
        status, data = self.call("GET", "/health")
        self.assertEqual((status, data["status"]), (200, "ok"))

    def test_auth_and_roles(self):
        txn = {"customer_id": "C1", "amount": 500}
        self.assertEqual(self.call("POST", "/score", txn)[0], 401)
        self.assertEqual(self.call("POST", "/score", txn, key="wrong")[0], 401)
        self.assertEqual(self.call("POST", "/score", txn, key="k-cust")[0], 200)
        self.assertEqual(self.call("GET", "/cases", key="k-cust")[0], 403)
        self.assertEqual(self.call("GET", "/metrics", key="k-admin")[0], 200)

    def test_rejected_request_does_not_break_keep_alive_connection(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        body = json.dumps({"customer_id": "CK", "amount": 100})
        c.request("POST", "/score", body, {"Content-Type": "application/json"})  # no key -> 401
        r = c.getresponse()
        r.read()
        self.assertEqual(r.status, 401)
        c.request("POST", "/score", body, {"Content-Type": "application/json", "X-API-Key": "k-cust"})
        r = c.getresponse()
        data = json.loads(r.read())
        c.close()
        self.assertEqual(r.status, 200)
        self.assertEqual(data["customer_id"], "CK")

    def test_validation_errors(self):
        for bad in ({"amount": 5}, {"customer_id": "C", "amount": -1}, {"customer_id": "C", "amount": "x"},
                    {"customer_id": "C", "amount": 5, "timestamp": "not-a-date"}):
            self.assertEqual(self.call("POST", "/score", bad, key="k-admin")[0], 422, bad)

    def test_suspicious_transaction_creates_case_that_analyst_can_resolve(self):
        # build a normal history, then a huge purchase from a new device in another city at 3am
        for i in range(8):
            self.call("POST", "/score", {"customer_id": "CX", "amount": 900, "city": "Salem", "device_id": "D1",
                                         "merchant_category": "grocery", "channel": "pos",
                                         "timestamp": f"2026-11-0{1 + i % 2}T1{i}:00:00"}, key="k-admin")
        status, res = self.call("POST", "/score", {
            "customer_id": "CX", "amount": 650000, "city": "Mumbai", "device_id": "UNSEEN",
            "merchant_category": "jewellery", "channel": "online", "timestamp": "2026-11-03T03:10:00"}, key="k-admin")
        self.assertEqual(status, 200)
        self.assertEqual(res["decision"], "block")  # hard rule: new device + amount >= 500,000
        self.assertIn("case_id", res)
        status, listing = self.call("GET", "/cases?status=open", key="k-admin")
        self.assertEqual(status, 200)
        self.assertTrue(any(c["id"] == res["case_id"] for c in listing["cases"]))
        status, case = self.call("POST", f"/cases/{res['case_id']}/resolve",
                                 {"status": "confirmed_fraud", "analyst": "tester"}, key="k-admin")
        self.assertEqual((status, case["status"]), (200, "confirmed_fraud"))
        self.assertTrue((self.root / "o.jsonl").read_text().strip())  # alert was written to the outbox


if __name__ == "__main__":
    unittest.main()
