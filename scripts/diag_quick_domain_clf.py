"""One-off diagnostic: how much worse is the 'quick' single-config domain
classifier (used as routing context in Tables 6/8/10/11) than the fully
grid-searched roberta-base from Table 5, on the identical seed-13 test split?
Not part of the experiment pipeline -- just answers a question before deciding
whether to change build_routing_domain_classifier.
"""
from experiments.common import build_encoders, load_and_split_amtcele, load_config, split_by
from domain.model import DomainClassifierModel
from domain.predict import predict_domain_batch
from eval.metrics import domain_report
from utils.types import DOMAINS

cfg = load_config("configs/default.yaml")
seed = cfg["seeds"][0]

semantic_encoder, _ = build_encoders(cfg)
items = load_and_split_amtcele(cfg, seed, semantic_encoder=semantic_encoder)
test_items = split_by(items, "test")

model = DomainClassifierModel.load("checkpoints/quick_domain_classifier/seed13")
preds = predict_domain_batch(model, [it.text for it in test_items])
report = domain_report(
    [it.domain for it in test_items],
    [p.predicted_domain for p in preds],
    [p.distribution for p in preds],
    [p.confidence for p in preds],
    DOMAINS,
)
print("quick_domain_classifier/seed13 on Table-5's seed-13 test split:")
print(f"  macro_f1={report['macro_f1']:.4f}  balanced_accuracy={report['balanced_accuracy']:.4f}  ece={report['ece']:.4f}")
print("Table 5's roberta-base (full grid search, same split): macro_f1=0.9627  balanced_accuracy=0.9643  ece=0.0206")
