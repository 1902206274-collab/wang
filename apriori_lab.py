import csv
import time
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

# Mac俄文字体，如果出现方框乱码就注释掉这一行
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.linewidth"] = 1.0


def load_dataset(file_path: str):
    transactions = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            items = [i.strip() for i in row if i.strip()]
            if items:
                transactions.append(items)
    return transactions


def create_C1(transactions):
    cnt = defaultdict(int)
    for tran in transactions:
        for item in tran:
            cnt[frozenset([item])] += 1
    return cnt


def filter_min_support(candidate, total_trans: int, min_sup_pct: float):
    min_count = min_sup_pct / 100.0 * total_trans
    freq = {}
    for itemset, count in candidate.items():
        if count >= min_count:
            freq[itemset] = count
    return freq


def apriori_generate(prev_freq: dict, k: int):
    itemsets = list(prev_freq.keys())
    candidates = defaultdict(int)
    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            a = sorted(list(itemsets[i]))
            b = sorted(list(itemsets[j]))
            if a[:k - 2] == b[:k - 2]:
                new_set = frozenset(itemsets[i] | itemsets[j])
                candidates[new_set] = 0
    return candidates


def count_candidates(candidates, transactions):
    counter = defaultdict(int)
    for tran in transactions:
        t_set = set(tran)
        for itemset in candidates:
            if itemset.issubset(t_set):
                counter[itemset] += 1
    return counter


def apriori(transactions, min_sup_pct: float):
    total = len(transactions)
    all_frequent = dict()
    c1 = create_C1(transactions)
    l1 = filter_min_support(c1, total, min_sup_pct)
    all_frequent.update(l1)
    current_L = l1
    k = 2
    while current_L:
        cand = apriori_generate(current_L, k)
        cand_cnt = count_candidates(cand, transactions)
        current_L = filter_min_support(cand_cnt, total, min_sup_pct)
        all_frequent.update(current_L)
        k += 1
    return all_frequent, total


def get_length_stat(freq_dict):
    stat = defaultdict(int)
    for s in freq_dict.keys():
        stat[len(s)] += 1
    return dict(stat)


def draw_itemset_tree(freq_sets):
    """修复版：弹簧布局，避免level属性缺失报错"""
    G = nx.DiGraph()

    # 取出现最多的6个1‑项集商品
    k1_all = [(s, cnt) for s, cnt in freq_sets.items() if len(s) == 1]
    k1_all.sort(key=lambda x: x[1], reverse=True)
    top_k1 = [itemset for itemset, _ in k1_all[:6]]

    # 最多8组高频2项集
    k2_candidate = []
    for s in freq_sets.keys():
        if len(s) == 2:
            items = list(s)
            if frozenset([items[0]]) in top_k1 and frozenset([items[1]]) in top_k1:
                k2_candidate.append((s, freq_sets[s]))
    k2_candidate.sort(key=lambda x: x[1], reverse=True)
    k2_filtered = [x[0] for x in k2_candidate[:8]]

    # 最多4组高频3项集
    k3_candidate = []
    for s in freq_sets.keys():
        if len(s) == 3:
            items = list(s)
            if all(frozenset([x]) in top_k1 for x in items):
                k3_candidate.append((s, freq_sets[s]))
    k3_candidate.sort(key=lambda x: x[1], reverse=True)
    k3_filtered = [x[0] for x in k3_candidate[:4]]

    # 添加一级节点
    for s in top_k1:
        label = ",".join(sorted(list(s)))
        G.add_node(label)

    # 添加二级节点
    for s in k2_filtered:
        label = ",".join(sorted(list(s)))
        G.add_node(label)
        items = sorted(list(s))
        for it in items:
            G.add_edge(it, label)

    # 添加三级节点
    for s in k3_filtered:
        label = ",".join(sorted(list(s)))
        G.add_node(label)
        items = sorted(list(s))
        for two_sub in combinations(items,2):
            fs_two = frozenset(two_sub)
            if fs_two in freq_sets:
                p_label = ",".join(two_sub)
                G.add_edge(p_label, label)

    plt.figure(figsize=(12,8),dpi=150)
    pos = nx.spring_layout(G, seed=42) #弹簧布局，seed固定保证每次图一样
    nx.draw_networkx(G, pos, with_labels=True, node_color="#a1caf1", node_size=2400,
                     font_size=8, arrowstyle="->",arrowsize=7)
    plt.title("Hierarchy of frequent itemsets (sample), min_sup=1%")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("tree_itemsets.png")
    plt.close()
    print("\n✅ Упрощенная диаграмма сохранена: tree_itemsets.png")


def draw_all_graphs(thresholds, time_list, k1_list, k2_list, k3_list):
    # 时间折线图
    plt.figure(figsize=(8, 5.2), dpi=150)
    plt.plot(thresholds, time_list, marker='o', linewidth=2.2, color="#1f77b4", markersize=6)
    plt.title("Зависимость времени выполнения от порога минимальной поддержки", fontsize=11)
    plt.xlabel("Порог минимальной поддержки, %", fontsize=10)
    plt.ylabel("Время выполнения, секунды", fontsize=10)
    plt.xticks([1, 3, 5, 10, 15])
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("plot_time_ru.png")
    plt.close()

    # 项集数量柱状图
    bar_width = 0.25
    x_pos = list(range(len(thresholds)))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    plt.figure(figsize=(10, 5.5), dpi=150)
    bar1 = plt.bar([p - bar_width for p in x_pos], k1_list, width=bar_width, color=colors[0], label="k=1")
    bar2 = plt.bar(x_pos, k2_list, width=bar_width, color=colors[1], label="k=2")
    bar3 = plt.bar([p + bar_width for p in x_pos], k3_list, width=bar_width, color=colors[2], label="k=3")

    def add_bar_label(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width() / 2., height + 1, f"{int(height)}",
                         ha="center", va="bottom", fontsize=8)

    add_bar_label(bar1)
    add_bar_label(bar2)
    add_bar_label(bar3)

    plt.xticks(x_pos, [f"{t}%" for t in thresholds])
    plt.title("Количество частых наборов при разных порогах поддержки", fontsize=11)
    plt.xlabel("Порог минимальной поддержки", fontsize=10)
    plt.ylabel("Количество частых наборов", fontsize=10)
    plt.legend(title="Длина набора")
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig("plot_k_ru.png")
    plt.close()
    print("\n✅ Диаграммы сохранены: plot_time_ru.png, plot_k_ru.png")


def main():
    parser = argparse.ArgumentParser(description="Эксперимент алгоритма Apriori")
    parser.add_argument("--file", required=True, help="Путь к файлу csv")
    parser.add_argument("--sup", type=float, help="Одиночный запуск: порог минимальной поддержки (%)")
    args = parser.parse_args()

    trans = load_dataset(args.file)

    if args.sup is not None:
        # 单独运行一个阈值
        sup = args.sup
        print(f"\n==== Расчет при пороге {sup}% ====")
        start_time = time.perf_counter()
        freq_sets, total_N = apriori(trans, sup)
        elapsed = time.perf_counter() - start_time
        len_stat = get_length_stat(freq_sets)
        k1 = len_stat.get(1, 0)
        k2 = len_stat.get(2, 0)
        k3 = len_stat.get(3, 0)

        print(f"Всего транзакций: {total_N}")
        print(f"Время работы: {elapsed:.4f} сек")
        print(f"Частых наборов всего: {len(freq_sets)}")
        print(f"k1={k1}, k2={k2}, k3={k3}")

        # 只有sup=1%才生成树状图
        if abs(sup - 1.0) < 1e-6:
            draw_itemset_tree(freq_sets)

    else:
        # 批量运行5组阈值，生成两张基础图表
        thresholds = [1, 3, 5, 10, 15]
        time_list = []
        k1_list = []
        k2_list = []
        k3_list = []
        for sup in thresholds:
            print(f"\n==== Расчет при пороге {sup}% ====")
            start_time = time.perf_counter()
            freq_sets, total_N = apriori(trans, sup)
            elapsed = time.perf_counter() - start_time
            len_stat = get_length_stat(freq_sets)
            k1 = len_stat.get(1, 0)
            k2 = len_stat.get(2, 0)
            k3 = len_stat.get(3, 0)

            time_list.append(elapsed)
            k1_list.append(k1)
            k2_list.append(k2)
            k3_list.append(k3)

            print(f"Всего транзакций: {total_N}")
            print(f"Время работы: {elapsed:.4f} сек")
            print(f"Частых наборов всего: {len(freq_sets)}")
            print(f"k1={k1}, k2={k2}, k3={k3}")
        draw_all_graphs(thresholds, time_list, k1_list, k2_list, k3_list)


if __name__ == "__main__":
    main()
