#!/usr/bin/env python
# coding=utf-8
from algorithm.DL.GCN2Cora import run_cora_classification


content_path = r"./data/cora/cora.content"
cites_path = r"./data/cora/cora.cites"
model, runtime, acc, nmi, mod = run_cora_classification(content_path, cites_path)