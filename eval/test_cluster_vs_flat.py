"""聚类索引 vs 全量FAISS — 100条记忆规模下的精度与速度对比。

问题: cluster-first search 在这个规模真的有用吗? 还是全量FAISS就够好?

设计:
  - 100条事实, 5个主题各20条 (programming / food / travel / fitness / finance)
  - 25条带 ground-truth 主题的查询
  - 对比三种检索:
      A) flat   — FactStore.retrieve, 预建FAISS, 只编码query
      B) cluster— ClusterIndex.search (现有实现, 每次重编码候选文本)
      C) cluster-opt — 聚类但预存embedding (剥离实现低效, 看算法本身)
  - 指标: top-1命中率 + top-5精度(同主题占比) + 每查询延迟
  - 以 flat 为参照: cluster是否丢结果? 是否更快?
"""

import sys
import shutil
import time
import statistics
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from memory_engine.fact_store import FactStore
from memory_engine.cluster_index import ClusterIndex

# ---- 100条事实: 5主题 x 20条 ----
TOPICS = {
    "programming": [
        "The user writes backend services in Python.",
        "The user prefers Go for high-concurrency systems.",
        "The user uses Rust for performance-critical modules.",
        "The user builds web frontends with TypeScript.",
        "The user maintains a legacy Java codebase.",
        "The user scripts automation tasks in Bash.",
        "The user deploys containers with Docker and Kubernetes.",
        "The user uses PostgreSQL as the primary database.",
        "The user caches data with Redis.",
        "The user writes unit tests with pytest.",
        "The user reviews pull requests on GitHub daily.",
        "The user debugs memory leaks with profilers.",
        "The user follows clean architecture principles.",
        "The user uses FastAPI to build REST endpoints.",
        "The user manages dependencies with Poetry.",
        "The user configures CI pipelines in GitHub Actions.",
        "The user writes async code with asyncio.",
        "The user instruments services with OpenTelemetry.",
        "The user prefers static typing over dynamic typing.",
        "The user refactors code to reduce cyclomatic complexity.",
    ],
    "food": [
        "The user loves eating fresh sushi.",
        "The user enjoys a bowl of spicy ramen.",
        "The user often orders Neapolitan pizza.",
        "The user grills burgers on weekends.",
        "The user likes Sichuan hotpot with friends.",
        "The user has dim sum every Sunday morning.",
        "The user cooks Thai green curry at home.",
        "The user snacks on street tacos.",
        "The user makes fresh pasta from scratch.",
        "The user slurps Vietnamese pho on cold days.",
        "The user bakes sourdough bread weekly.",
        "The user prefers dark roast coffee.",
        "The user drinks matcha green tea in the afternoon.",
        "The user enjoys dark chocolate desserts.",
        "The user eats a vegetarian diet on weekdays.",
        "The user ferments his own kimchi.",
        "The user loves a medium-rare ribeye steak.",
        "The user orders extra cheese on everything.",
        "The user picks mango as a favorite fruit.",
        "The user avoids overly sweet sodas.",
    ],
    "travel": [
        "The user backpacked across Southeast Asia.",
        "The user visited the temples of Kyoto.",
        "The user hiked the Inca Trail to Machu Picchu.",
        "The user took a road trip along the California coast.",
        "The user explored the canals of Venice.",
        "The user went on safari in Kenya.",
        "The user climbed base camp at Everest.",
        "The user toured the museums of Paris.",
        "The user dove the Great Barrier Reef.",
        "The user wandered the markets of Marrakech.",
        "The user cruised through the Norwegian fjords.",
        "The user watched the northern lights in Iceland.",
        "The user rode trains across Switzerland.",
        "The user camped in the Patagonian wilderness.",
        "The user surfed the beaches of Bali.",
        "The user strolled the old town of Prague.",
        "The user trekked through the Scottish Highlands.",
        "The user sailed around the Greek islands.",
        "The user visited the pyramids of Giza.",
        "The user explored the streets of Tokyo at night.",
    ],
    "fitness": [
        "The user runs five kilometers every morning.",
        "The user lifts weights at the gym three times a week.",
        "The user practices yoga to improve flexibility.",
        "The user swims laps at the local pool.",
        "The user cycles long distances on weekends.",
        "The user does high-intensity interval training.",
        "The user tracks daily steps with a fitness watch.",
        "The user stretches before every workout.",
        "The user trains for a marathon this fall.",
        "The user climbs at an indoor bouldering gym.",
        "The user follows a strict protein-rich diet.",
        "The user takes rest days to recover muscles.",
        "The user does pull-ups and push-ups at home.",
        "The user joined a local soccer league.",
        "The user practices meditation for mental fitness.",
        "The user uses a foam roller after running.",
        "The user monitors his resting heart rate.",
        "The user does squats to build leg strength.",
        "The user plays tennis on Thursday evenings.",
        "The user hikes hills to build endurance.",
    ],
    "finance": [
        "The user invests monthly in index funds.",
        "The user tracks expenses with a budgeting app.",
        "The user holds a diversified stock portfolio.",
        "The user contributes to a retirement account.",
        "The user pays off credit card balances in full.",
        "The user keeps an emergency fund of six months.",
        "The user reads quarterly earnings reports.",
        "The user owns some government bonds.",
        "The user avoids high-interest consumer debt.",
        "The user rebalances his portfolio every year.",
        "The user dollar-cost-averages into the market.",
        "The user uses a tax-advantaged savings plan.",
        "The user compares mortgage interest rates.",
        "The user allocates a small amount to crypto.",
        "The user reviews his net worth each month.",
        "The user negotiates his salary at reviews.",
        "The user buys dividend-paying stocks.",
        "The user keeps savings in a high-yield account.",
        "The user studies compound interest carefully.",
        "The user plans for early retirement.",
    ],
}

# ---- 25条查询, 每条标注期望主题 ----
QUERIES = [
    ("What programming language does the user use for backend?", "programming"),
    ("How does the user deploy applications?", "programming"),
    ("What database does the user rely on?", "programming"),
    ("How does the user test code?", "programming"),
    ("What does the user do for CI/CD?", "programming"),
    ("What kind of food does the user enjoy?", "food"),
    ("Does the user like spicy dishes?", "food"),
    ("What does the user drink in the morning?", "food"),
    ("What desserts does the user prefer?", "food"),
    ("Is the user vegetarian?", "food"),
    ("Where has the user traveled in Asia?", "travel"),
    ("Did the user go hiking on any trips?", "travel"),
    ("What did the user see in Europe?", "travel"),
    ("Has the user been on a safari?", "travel"),
    ("What water activities did the user do?", "travel"),
    ("How does the user stay fit?", "fitness"),
    ("Does the user run regularly?", "fitness"),
    ("What does the user do at the gym?", "fitness"),
    ("Is the user training for a race?", "fitness"),
    ("How does the user track workouts?", "fitness"),
    ("How does the user invest money?", "finance"),
    ("How does the user manage debt?", "finance"),
    ("Does the user save for retirement?", "finance"),
    ("What does the user do with stocks?", "finance"),
    ("How does the user budget expenses?", "finance"),
]


def topic_of(text: str) -> str:
    for topic, facts in TOPICS.items():
        if text in facts:
            return topic
    return "?"


class ClusterOpt:
    """聚类索引(优化版): 预存所有记忆embedding, 检索时不重编码候选。

    剥离现有 ClusterIndex 每次重编码候选文本的实现低效,
    看 cluster-first 算法本身在这个规模的速度上限。
    """

    def __init__(self, fact_store, n_clusters=5):
        self.facts = fact_store
        self.n_clusters = n_clusters
        self._emb = None
        self._centroids = None
        self._cluster_indices = None

    def build(self):
        self.facts._ensure_model()
        texts = [f["text"] for f in self.facts.facts]
        self._emb = np.array(self.facts._model.encode(texts, normalize_embeddings=True), dtype=np.float32)
        from sklearn.cluster import KMeans
        k = min(self.n_clusters, len(texts) // 3)
        km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=50)
        labels = km.fit_predict(self._emb)
        c = km.cluster_centers_
        self._centroids = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
        self._cluster_indices = {}
        for i, l in enumerate(labels):
            self._cluster_indices.setdefault(int(l), []).append(i)
        return k

    def search(self, query, top_k=5, n_probe=2):
        q = np.array(self.facts._model.encode([query], normalize_embeddings=True), dtype=np.float32)
        csim = (q @ self._centroids.T)[0]
        top_clusters = np.argsort(csim)[::-1][:n_probe]
        cand = []
        for cid in top_clusters:
            cand.extend(self._cluster_indices.get(int(cid), []))
        cand = np.array(cand)
        sims = (q @ self._emb[cand].T)[0]
        order = np.argsort(sims)[::-1][:top_k]
        return [self.facts.facts[cand[o]] for o in order]


def precision_at_5(results, expected_topic):
    top5 = results[:5]
    if not top5:
        return 0.0
    return sum(1 for r in top5 if topic_of(r["text"]) == expected_topic) / len(top5)


def top1_hit(results, expected_topic):
    if not results:
        return 0
    return 1 if topic_of(results[0]["text"]) == expected_topic else 0


def bench(name, search_fn, warmup_query):
    """跑全部查询, 返回 (top1命中率, 平均p@5, 中位延迟ms, 均值延迟ms)。"""
    search_fn(warmup_query)  # warmup (排除首调用开销)
    top1, p5, lat = [], [], []
    per_query = []
    for q, expected in QUERIES:
        t0 = time.perf_counter()
        res = search_fn(q)
        dt = (time.perf_counter() - t0) * 1000
        lat.append(dt)
        h = top1_hit(res, expected)
        p = precision_at_5(res, expected)
        top1.append(h)
        p5.append(p)
        per_query.append((q, expected, h, p, dt))
    return {
        "name": name,
        "top1": sum(top1) / len(top1),
        "p5": sum(p5) / len(p5),
        "lat_median": statistics.median(lat),
        "lat_mean": statistics.mean(lat),
        "lat_p90": sorted(lat)[int(len(lat) * 0.9)],
        "per_query": per_query,
    }


def main():
    print("=" * 64)
    print("  Cluster-First vs Flat FAISS  (100 facts, 5 topics)")
    print("=" * 64)

    shutil.rmtree("./_cvf_test", ignore_errors=True)
    fs = FactStore("./_cvf_test")

    print("\n[setup] 写入100条事实(5主题x20)...", flush=True)
    t0 = time.perf_counter()
    # 批量加载: 直接append再一次性建索引(避免每次add都O(n)重编码 -> O(n^2))
    now = int(time.time())
    for topic, facts in TOPICS.items():
        for text in facts:
            fs.facts.append({"id": fs._next_id, "text": text, "pinned": False,
                             "ts": now, "trust": 0.5, "uses": 0, "last_used": now})
            fs._next_id += 1
    fs._save()
    fs._rebuild_index()  # 一次性编码全部100条
    print(f"  写入完成: {len(fs.facts)}条, 耗时 {time.perf_counter()-t0:.1f}s", flush=True)

    # 构建聚类索引(现有实现)
    print("\n[setup] 构建聚类索引...")
    ci = ClusterIndex(fs, n_clusters=5)
    t0 = time.perf_counter()
    cres = ci.build()
    print(f"  ClusterIndex.build: {cres['status']}, k={cres.get('n_clusters')}, "
          f"sizes={cres.get('cluster_sizes')}, {time.perf_counter()-t0:.1f}s")

    # 构建优化版
    cop = ClusterOpt(fs, n_clusters=5)
    cop.build()

    # 看聚类纯度: 每个cluster里主导主题占比
    print("\n[聚类纯度] 每个cluster的主题分布:")
    for cid, indices in sorted(ci._cluster_indices.items()):
        topics = [topic_of(fs.facts[i]["text"]) for i in indices]
        from collections import Counter
        dist = Counter(topics)
        dominant, dn = dist.most_common(1)[0]
        print(f"  cluster {cid}: {len(indices)}条, 主导={dominant}({dn}/{len(indices)}="
              f"{dn/len(indices)*100:.0f}%), 分布={dict(dist)}")

    # ---- 三种检索基准 ----
    runs = []
    runs.append(bench("flat (FAISS)", lambda q: fs.retrieve(q, top_k=5),
                      "warmup programming query"))
    runs.append(bench("cluster (as-impl)", lambda q: ci.search(q, top_k=5, n_probe_clusters=2),
                      "warmup programming query"))
    runs.append(bench("cluster-opt (precomp)", lambda q: cop.search(q, top_k=5, n_probe=2),
                      "warmup programming query"))

    # ---- 报告 ----
    print("\n" + "=" * 64)
    print("  结果")
    print("=" * 64)
    print(f"\n{'方法':<24} {'top1命中':>9} {'p@5':>8} {'延迟中位':>10} {'延迟均值':>10} {'p90':>9}")
    print("-" * 74)
    for r in runs:
        print(f"{r['name']:<24} {r['top1']*100:>7.1f}% {r['p5']*100:>7.1f}% "
              f"{r['lat_median']:>8.1f}ms {r['lat_mean']:>8.1f}ms {r['lat_p90']:>7.1f}ms")

    # ---- flat vs cluster 逐查询差异 ----
    flat = runs[0]
    clus = runs[1]
    print("\n[逐查询: flat vs cluster(as-impl) 差异]")
    diffs = 0
    for (q, ex, h_f, p_f, _), (_, _, h_c, p_c, _) in zip(flat["per_query"], clus["per_query"]):
        if p_f != p_c or h_f != h_c:
            diffs += 1
            print(f"  '{q[:45]}' ({ex})")
            print(f"      flat: top1={h_f} p@5={p_f:.2f}  |  cluster: top1={h_c} p@5={p_c:.2f}")
    if diffs == 0:
        print("  无差异: cluster与flat在所有25条查询上返回相同质量的结果")

    # ---- 速度对比结论 ----
    print("\n[速度对比]")
    f_lat = flat["lat_median"]
    for r in runs[1:]:
        ratio = r["lat_median"] / f_lat
        verdict = f"{ratio:.2f}x flat ({'更慢' if ratio > 1 else '更快'})"
        print(f"  {r['name']:<24}: {verdict}")

    # ---- 结论 ----
    print("\n" + "=" * 64)
    print("  结论")
    print("=" * 64)
    acc_loss = (flat["p5"] - clus["p5"]) * 100
    print(f"  - 精度: flat p@5={flat['p5']*100:.1f}%, cluster p@5={clus['p5']*100:.1f}% "
          f"({'cluster丢' if acc_loss>0 else 'cluster多' if acc_loss<0 else '持平'}{abs(acc_loss):.1f}pt)")
    print(f"  - 速度: cluster(as-impl)={clus['lat_median']/f_lat:.2f}x, "
          f"cluster-opt={runs[2]['lat_median']/f_lat:.2f}x (相对flat)")
    print(f"  - 100条规模: flat单查询仅 {f_lat:.1f}ms")

    import json
    out = {
        "scale": len(fs.facts),
        "n_queries": len(QUERIES),
        "cluster_build": cres,
        "runs": [{k: v for k, v in r.items() if k != "per_query"} for r in runs],
    }
    Path("./eval/cluster_vs_flat_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n  结果已存 eval/cluster_vs_flat_results.json")


if __name__ == "__main__":
    main()
