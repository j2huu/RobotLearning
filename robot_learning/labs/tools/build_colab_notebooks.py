"""Build standalone teaching notebooks. This builds files; it never runs experiments.

Run from any directory with Python's standard library. Generated notebooks embed
all runtime code and synthetic data generators; they do not import this builder.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import textwrap

LABS = Path(__file__).resolve().parents[1]
OUT = LABS / "notebooks"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n",
            "execution_count": None, "outputs": []}


def start(title, goal, torch=False, sklearn=False):
    imports = ""
    versions = ""
    if torch:
        imports += "import torch\nfrom torch import nn\nimport torch.nn.functional as F\ntorch.manual_seed(SEED)\ntorch.set_num_threads(2)\n"
        versions += "print('torch:', torch.__version__, '| teaching device: CPU')\n"
    if sklearn:
        imports += "import sklearn\n"
        versions += "print('sklearn:', sklearn.__version__)\n"
    return [md(f"""
        # {title}

        {goal}

        **运行方式**：上传本 `.ipynb` 到 Google Colab，连接 Python runtime，从上到下运行。
        默认 CPU 即可；本 notebook 自带代码和合成数据，不需要克隆仓库、上传论文、挂载 Drive 或配置本地环境。
        所有输出留空，首次运行后才产生实际结果；不要把材料准备状态当作已复现实验。

        如果所选云端 runtime 缺少包，可在 Colab 新单元执行 `%pip install numpy matplotlib pandas scikit-learn torch`，
        然后重启 runtime。不要为了本课程在本机安装依赖。
        """), md("## 1. 环境与随机性\n记录版本和 seed。换 seed 时从本单元重新运行；图中用英文标签避免依赖额外中文字体。"),
        code(textwrap.dedent("""
        import json
        import math
        import platform
        import random
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        SEED = 7
        random.seed(SEED)
        np.random.seed(SEED)
        rng = np.random.default_rng(SEED)
        print('Python:', platform.python_version(), '| NumPy:', np.__version__, '| seed:', SEED)
        """) + imports + versions)]


def finish(cells, prompts):
    cells += [md("## 面试追问与实验记录\n" + prompts + "\n\n记录 Question / Data / Config / Metric / Observation / Explanation / Limitation。\n改变实验设置时不要反复用 test 选择答案；保存 notebook 后再下载 `.ipynb` 保留输出。")]
    cells += [md("## 可选：导出指标\n默认关闭。需要时打开开关；文件只写到当前 **Colab runtime** 的 `outputs/`。runtime 临时文件不等于已永久保存，可使用 Colab 文件面板下载。"), code("""
        EXPORT_RESULTS = False
        if EXPORT_RESULTS:
            destination = Path('outputs')
            destination.mkdir(exist_ok=True)
            (destination / 'metrics.json').write_text(
                json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            print('Saved outputs/metrics.json in the Colab runtime; download it from the Files panel.')
        """)]
    return cells


def save(name, cells):
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{name}:{index}".encode()).hexdigest()[:12]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": name, "provenance": []},
    }, "nbformat": 4, "nbformat_minor": 5}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def embedded_definitions(filename):
    """Copy original definitions as notebook cells; omit CLI main and __file__ use."""
    source = (LABS / filename).read_text()
    tree = ast.parse(source)
    imports, definitions = [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import) and any(a.name == "argparse" for a in node.names):
                continue
            imports.append(ast.get_source_segment(source, node))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "main":
            # get_source_segment excludes decorators, whose start precedes node.lineno.
            first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            definitions.append((node.name, "\n".join(source.splitlines()[first:node.end_lineno])))
    return [code("\n".join(imports))] + [cell for name, definition in definitions
        for cell in (md(f"### 定义：`{name}`\n先看输入输出和 shape，再运行定义。本单元不启动训练。"), code(definition))]


def original_bc():
    cells = start("Lab01：概率 BC 与多峰动作", "比较 MSE、单 Gaussian 与 mixture：低 mean MSE 是否代表能采到合理模式？", torch=True)
    cells += [md("## 2. 自包含模型与数据定义\n以下定义迁移自原 `01_probabilistic_bc.py`，不读取该脚本。隐藏 branch 不输入 policy，split 按 episode。")]
    cells += embedded_definitions("01_probabilistic_bc.py")
    cells += [md("## 3. 配置、训练与 held-out 评价\n三模型使用同一数据与 batch/steps。NLL 只在概率模型间有定义；mode compatibility 是合成模式距离，不是机器人成功率。"), code("""
        STEPS = 600
        BATCH_SIZE = 128
        train_obs, train_action, val_obs, val_action, metadata = make_data(SEED)
        assert set(metadata['train_episode_ids']).isdisjoint(metadata['validation_episode_ids'])
        policies, traces, metrics = {}, {}, {}
        for index, kind in enumerate(('mse', 'gaussian', 'mixture')):
            torch.manual_seed(SEED + 100 * index)
            policy = Policy(kind)
            optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
            history = []
            policy.train()
            for step in range(STEPS):
                indices = torch.randint(len(train_obs), (BATCH_SIZE,))
                objective = loss(policy, train_obs[indices], train_action[indices])
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
                if step % 10 == 0:
                    history.append((step, float(objective.detach())))
            policies[kind], traces[kind] = policy, history
            metrics[kind] = evaluate(policy, val_obs, val_action)
        results = {'seed': SEED, 'steps': STEPS, 'data': metadata, 'models': metrics,
                   'scope': 'Synthetic conditional distribution; no robot rollout; validation split only.'}
        display(pd.DataFrame(metrics).T.drop(columns='distribution_at_o_zero'))
        """), md("## 4. 观察 o=0 的采样分布\n模型的 mean 与 sample 分开看。两条 expert modes 在 -1/+1；单 Gaussian 的均值可能仍在中间。"), code("""
        fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=True)
        with torch.no_grad():
            for axis, (kind, policy) in zip(axes, policies.items()):
                samples = sample_actions(policy, torch.zeros(4000, 1)).numpy().ravel()
                axis.hist(samples, bins=60, range=(-2, 2), density=True, alpha=0.7)
                axis.axvline(-1, color='black', linestyle='--')
                axis.axvline(1, color='black', linestyle='--')
                axis.set(title=kind, xlabel='action at o=0')
        axes[0].set_ylabel('density')
        plt.tight_layout(); plt.show()
        """)]
    return finish(cells, "1. 推导固定方差 Gaussian NLL 与 MSE 的关系。\n2. 为什么 mixture 的 mean 仍可能在两个 modes 之间？\n3. 固定数据，仅减少 steps；解释优化不充分与分布假设错误的区别。")


def original_windows():
    cells = start("Lab02：episode windows、mask 与因果对齐", "从合成 episodes 构造 history/action chunks，检查未来 force 泄漏与 train-only normalization。", torch=True)
    cells += embedded_definitions("02_episode_windows.py")
    source = (LABS / "02_episode_windows.py").read_text()
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = textwrap.dedent("\n".join(source.splitlines()[main.lineno:main.end_lineno]))
    body = body[body.index("torch.set_num_threads"):body.index("print(json.dumps(result")]
    anchors = ["train_actions =", "observation_horizon,", "normalized =", "result ="]
    positions = [0] + [body.index(a) for a in anchors] + [len(body)]
    descriptions = ["## 2. 生成并按 episode split", "## 3. 只用真实 train actions 拟合统计",
                    "## 4. 构造 windows 与 padding masks", "## 5. Normalization round trip 与 masked loss",
                    "## 6. 汇总检查与边界样本"]
    for title, begin, end in zip(descriptions, positions, positions[1:]):
        cells += [md(title), code(body[begin:end])]
    cells += [code("""
        results = result
        assert result['future_force_sample_count'] == 0
        print(json.dumps(results, indent=2))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].imshow(observation_mask[:12].numpy(), aspect='auto', vmin=0, vmax=1)
        axes[0].set(title='Observation valid mask', xlabel='history slot', ylabel='window')
        axes[1].imshow(action_mask[-12:].numpy(), aspect='auto', vmin=0, vmax=1)
        axes[1].set(title='Action valid mask', xlabel='future slot', ylabel='window')
        plt.tight_layout(); plt.show()
        """)]
    return finish(cells, "1. 为什么统计不能从重叠 windows 重复计数、也不能读取 validation？\n2. 把对齐规则改成最近时间点，何时会用到未来 force？\n3. padding loss 的分母应该是什么？本实验没有 image pipeline 或真实 frame conversion。")


def original_pi():
    cells = start("Pi 机制：flow 符号、两层 attention 泄漏与时延", "保留原标准库机制检查：不训练 Transformer，不使用真实 RFM，也不启动物理模拟。")
    cells += embedded_definitions("pi_series_mechanics.py")
    cells += [md("## 2. 运行三个机制检查\nFlow 部分使用已知端点的 oracle velocity；真正学习 velocity field 的实验见 ML06。"), code("""
        STEPS = 10
        results = {'seed': SEED, 'steps': STEPS, 'flow': check_flow(SEED, STEPS),
                   'attention': check_attention(), 'delay': check_delay(),
                   'scope': 'Mechanism checks only, not learned-model or robot validation.'}
        print(json.dumps(results, indent=2))
        """), md("## 3. 显示允许的 attention 依赖与异步预算"), code("""
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        mask_array = np.asarray(results['attention']['correct_mask'])
        axes[0].imshow(mask_array, vmin=0, vmax=1)
        labels = results['attention']['token_order']
        axes[0].set_xticks(range(len(labels)), labels, rotation=45)
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].set(title='Allowed query-to-key dependencies', xlabel='key', ylabel='query')
        cases = results['delay']['cases']
        axes[1].bar([str(c['latency_ms']) for c in cases],
                    [c['async_budget_margin_ms_ignoring_other_overheads'] for c in cases])
        axes[1].axhline(0, color='black')
        axes[1].set(title='200 ms overlap budget', xlabel='inference latency (ms)', ylabel='margin (ms)')
        plt.tight_layout(); plt.show()
        """)]
    return finish(cells, "1. 为什么只翻 velocity target、不翻积分方向会错？\n2. 为什么第一层没有泄漏、第二层却能泄漏？\n3. 1 kHz 插值点、50 Hz policy 与 inference latency 是三个不同量。")


def ml01():
    cells = start("ML01：手写线性/逻辑回归与正则化", "从 NumPy loss/gradient 到 validation 模型选择，避免只会调用 fit。", sklearn=True)
    cells += [md("## 2. 手写 linear MSE gradient 与有限差分\n使用半 MSE，使梯度为 X.T@(Xw-y)/N。y 与 prediction 都是 [N,1]。"), code("""
        X = rng.normal(size=(400, 3)) * np.array([1., 8., 0.05])
        y = X @ np.array([[1.], [0.3], [-2.]]) + rng.normal(scale=0.2, size=(400, 1))
        train_ids, val_ids = np.arange(280), np.arange(280, 400)
        Xtr, Xva, ytr, yva = X[train_ids], X[val_ids], y[train_ids], y[val_ids]
        def linear_loss_grad(x, target, weights):
            prediction = x @ weights
            assert prediction.shape == target.shape
            error = prediction - target
            return float(0.5 * np.mean(error**2)), x.T @ error / len(x)
        w = rng.normal(size=(3, 1))
        objective, analytic = linear_loss_grad(Xtr, ytr, w)
        numerical = np.zeros_like(w)
        eps = 1e-5
        for j in range(len(w)):
            delta = np.zeros_like(w); delta[j, 0] = eps
            numerical[j, 0] = (linear_loss_grad(Xtr, ytr, w+delta)[0] -
                               linear_loss_grad(Xtr, ytr, w-delta)[0]) / (2*eps)
        assert np.allclose(analytic, numerical, rtol=1e-4, atol=1e-5)
        print('linear gradient max error:', np.max(np.abs(analytic-numerical)))
        """), md("## 3. 同样 LR/steps，仅改变 feature scale\nScaler 只 fit train；intercept 用额外一列 1 表示。图是优化轨迹，不是统计泛化胜负。"), code("""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(Xtr)
        histories = {}
        for name, x_train, x_valid in [('raw', Xtr, Xva),
                                      ('standardized', scaler.transform(Xtr), scaler.transform(Xva))]:
            a = np.column_stack([x_train, np.ones(len(x_train))])
            b = np.column_stack([x_valid, np.ones(len(x_valid))])
            weights = np.zeros((a.shape[1], 1))
            trace = []
            for step in range(1000):
                value, gradient = linear_loss_grad(a, ytr, weights)
                weights -= 0.01 * gradient
                if step % 10 == 0:
                    trace.append((step, value, linear_loss_grad(b, yva, weights)[0]))
            histories[name] = np.asarray(trace)
        for name, trace in histories.items():
            plt.plot(trace[:, 0], trace[:, 2], label=name)
        plt.yscale('log'); plt.xlabel('optimizer steps'); plt.ylabel('validation half-MSE')
        plt.legend(); plt.show()
        """), md("## 4. 手写 logistic：stable BCE 与 p-y 梯度"), code("""
        Xc = rng.normal(size=(700, 3))
        Xc = np.column_stack([Xc, np.ones(len(Xc))])
        p_true = 1 / (1 + np.exp(-(Xc @ np.array([1.5, -1., .5, -.2]))))
        yc = (rng.random(len(Xc)) < p_true).astype(float).reshape(-1, 1)
        wc = np.zeros((4, 1)); logistic_trace = []
        for step in range(500):
            logits = Xc[:450] @ wc
            probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
            value = np.mean(np.logaddexp(0, logits) - yc[:450] * logits)
            gradient = Xc[:450].T @ (probabilities-yc[:450]) / 450
            wc -= 0.1 * gradient
            logistic_trace.append(float(value))
        held_logits = Xc[450:] @ wc
        held_loss = float(np.mean(np.logaddexp(0, held_logits)-yc[450:]*held_logits))
        print('held-out logistic NLL:', held_loss)
        plt.plot(logistic_trace); plt.xlabel('step'); plt.ylabel('training BCE'); plt.show()
        """), md("## 5. Polynomial + Ridge：先 validation 选择，再 test\n以下是另一份独立合成回归数据。test 不参与 degree/penalty 选择。"), code("""
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        xp = rng.uniform(-1, 1, size=(160, 1))
        yp = np.sin(3*xp[:, 0]) + rng.normal(scale=0.2, size=160)
        order = rng.permutation(160); tr, va, te = order[:80], order[80:120], order[120:]
        candidates, rows = {}, []
        for degree in (1, 3, 9, 15):
            for penalty in (0.001, 1., 50.):
                model = make_pipeline(PolynomialFeatures(degree, include_bias=False),
                                      StandardScaler(), Ridge(alpha=penalty))
                model.fit(xp[tr], yp[tr])
                candidates[(degree, penalty)] = model
                rows.append({'degree': degree, 'penalty': penalty,
                             'train_mse': mean_squared_error(yp[tr], model.predict(xp[tr])),
                             'val_mse': mean_squared_error(yp[va], model.predict(xp[va]))})
        table = pd.DataFrame(rows).sort_values('val_mse'); display(table)
        best = table.iloc[0]; key = (int(best['degree']), float(best['penalty']))
        selected = candidates[key]
        test_mse = float(mean_squared_error(yp[te], selected.predict(xp[te])))
        results = {'seed': SEED, 'gradient_max_error': float(np.max(np.abs(analytic-numerical))),
                   'logistic_held_out_nll': held_loss, 'selected_degree': key[0],
                   'selected_penalty': key[1], 'selected_test_mse': test_mse}
        print(results)
        grid = np.linspace(-1, 1, 200).reshape(-1, 1)
        plt.scatter(xp[tr, 0], yp[tr], s=14, label='train')
        plt.plot(grid[:, 0], selected.predict(grid), label='validation-selected model')
        plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.show()
        """)]
    return finish(cells, "1. 手推 p-y；为什么 BCE 接 logits 更稳定？\n2. 只改变 degree/penalty，解释 bias/variance。\n3. 为什么在 validation 上挑最小值后还需要独立 test？")


def ml02():
    cells = start("ML02：经典模型、不平衡指标与数据泄漏", "比较合理 baselines，严格区分 model selection、threshold selection 与 test evaluation。", sklearn=True)
    cells += [md("## 2. 固定 train/validation/test 与候选模型\n合成 tabular classification，不是机器人 rollout。所有 preprocessing 都放在 Pipeline 内。"), code("""
        from time import perf_counter
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split, GroupShuffleSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
        from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
            precision_score, recall_score, accuracy_score, brier_score_loss,
            confusion_matrix, precision_recall_curve, roc_curve)
        from sklearn.calibration import calibration_curve
        X, y = make_classification(n_samples=1200, n_features=12, n_informative=6,
            n_redundant=2, weights=[.9, .1], flip_y=.02, random_state=SEED)
        ids = np.arange(len(y))
        train_ids, held_ids = train_test_split(ids, test_size=.4, stratify=y, random_state=SEED)
        val_ids, test_ids = train_test_split(held_ids, test_size=.5, stratify=y[held_ids], random_state=SEED+1)
        assert set(train_ids).isdisjoint(val_ids) and set(held_ids).isdisjoint(train_ids)
        models = {
            'logistic': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
            'svm_rbf': make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
            'tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEED),
            'random_forest': RandomForestClassifier(n_estimators=80, min_samples_leaf=3,
                                                     random_state=SEED, n_jobs=1),
            'boosting': HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, random_state=SEED),
        }
        rows = []
        for name, model in models.items():
            begin = perf_counter(); model.fit(X[train_ids], y[train_ids])
            rows.append({'model': name, 'fit_seconds': perf_counter()-begin,
                         'val_ap': average_precision_score(y[val_ids], model.predict_proba(X[val_ids])[:, 1])})
        selection = pd.DataFrame(rows).sort_values('val_ap', ascending=False)
        display(selection)
        selected_name = str(selection.iloc[0]['model']); selected = models[selected_name]
        """), md("## 3. Validation 选 threshold；只评价选定模型的 test\n这里允许 validation 同时选择模型和阈值，因此最终必须用独立 test。"), code("""
        val_prob = selected.predict_proba(X[val_ids])[:, 1]
        thresholds = np.linspace(.05, .95, 91)
        val_f1 = [f1_score(y[val_ids], val_prob >= t, zero_division=0) for t in thresholds]
        threshold = float(thresholds[int(np.argmax(val_f1))])
        test_prob = selected.predict_proba(X[test_ids])[:, 1]
        prediction = test_prob >= threshold
        test_y = y[test_ids]
        results = {'seed': SEED, 'selected_model': selected_name, 'validation_selected_threshold': threshold,
            'test_n': len(test_ids), 'test_prevalence': float(test_y.mean()),
            'test_accuracy': float(accuracy_score(test_y, prediction)),
            'test_precision': float(precision_score(test_y, prediction, zero_division=0)),
            'test_recall': float(recall_score(test_y, prediction, zero_division=0)),
            'test_f1': float(f1_score(test_y, prediction, zero_division=0)),
            'test_ap': float(average_precision_score(test_y, test_prob)),
            'test_roc_auc': float(roc_auc_score(test_y, test_prob)),
            'test_brier': float(brier_score_loss(test_y, test_prob)),
            'always_negative_accuracy': float((test_y == 0).mean()),
            'confusion_matrix': confusion_matrix(test_y, prediction).tolist()}
        print(json.dumps(results, indent=2))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        precision, recall, _ = precision_recall_curve(test_y, test_prob)
        axes[0].plot(recall, precision)
        axes[0].axhline(test_y.mean(), color='black', linestyle='--')
        axes[0].set(xlabel='recall', ylabel='precision', title='Test PR curve')
        fpr, tpr, _ = roc_curve(test_y, test_prob)
        axes[1].plot(fpr, tpr); axes[1].plot([0, 1], [0, 1], '--', color='black')
        axes[1].set(xlabel='FPR', ylabel='TPR', title='Test ROC curve')
        fraction, mean_prob = calibration_curve(test_y, test_prob, n_bins=6)
        axes[2].plot(mean_prob, fraction, 'o-'); axes[2].plot([0,1],[0,1],'--',color='black')
        axes[2].set(xlabel='mean probability', ylabel='positive fraction', title='Reliability diagram')
        plt.tight_layout(); plt.show()
        """), md("## 4. 固定规则的 bootstrap 区间\n此合成数据中每行独立，按行 bootstrap；真实机器人重复 rollout/episode 内的 frames 不满足这个假设。"), code("""
        bootstrap_rng = np.random.default_rng(SEED+100)
        scores = []
        for _ in range(400):
            sample = bootstrap_rng.integers(0, len(test_y), size=len(test_y))
            scores.append(f1_score(test_y[sample], prediction[sample], zero_division=0))
        results['test_f1_bootstrap_95_percentile_interval'] = np.quantile(scores, [.025, .975]).tolist()
        print('F1 percentile interval:', results['test_f1_bootstrap_95_percentile_interval'])
        """), md("## 5. 独立反例：frame split 记住 group ID\n下面不是重新调上面的 test。每组 label 随机、组内共享 ID；新组没有可泛化信号，1NN 可记住见过的组。"), code("""
        group_rng = np.random.default_rng(SEED+10)
        n_groups, per_group = 100, 8
        groups = np.repeat(np.arange(n_groups), per_group)
        group_labels = group_rng.integers(0, 2, size=n_groups)
        gy = group_labels[groups]
        gx = np.eye(n_groups)[groups]
        frame_train, frame_eval = train_test_split(np.arange(len(gy)), test_size=.3, random_state=SEED)
        group_train, group_eval = next(GroupShuffleSplit(n_splits=1, test_size=.3, random_state=SEED).split(gx, gy, groups))
        leakage_rows = []
        for name, tr, ev in [('frame split', frame_train, frame_eval), ('group split', group_train, group_eval)]:
            memorizer = KNeighborsClassifier(n_neighbors=1).fit(gx[tr], gy[tr])
            leakage_rows.append({'split': name, 'accuracy': float(accuracy_score(gy[ev], memorizer.predict(gx[ev]))),
                                 'overlap_groups': len(set(groups[tr]) & set(groups[ev]))})
        results['group_identity_counterexample'] = leakage_rows
        display(pd.DataFrame(leakage_rows))
        """)]
    return finish(cells, "1. 为什么 always-negative accuracy 可能高，但 recall 为零？\n2. AP、ROC-AUC、threshold F1 与 Brier 各看什么？\n3. group split 的独立单位是什么？\n4. 小样本 bootstrap 的区间为什么不是普遍保证？")


def ml03():
    cells = start("ML03：autograd、训练诊断与 LoRA 梯度", "从手推 CE 梯度到 optimizer、train/eval、freeze 和 LoRA 初始化。", torch=True)
    cells += [md("## 2. CE 梯度：手算、autograd、有限差分\nlogits=[B,C]、整数 targets=[B]。Loss 对 batch mean，因此梯度除以 B。"), code("""
        logits = torch.tensor([[.3, -.2, .8], [.1, .7, -.4]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([2, 1])
        objective = F.cross_entropy(logits, targets)
        objective.backward()
        hand = (logits.detach().softmax(-1) - F.one_hot(targets, 3)) / len(targets)
        eps = 1e-5; numerical = torch.zeros_like(logits)
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                delta = torch.zeros_like(logits); delta[i, j] = eps
                numerical[i, j] = (F.cross_entropy(logits.detach()+delta, targets) -
                                    F.cross_entropy(logits.detach()-delta, targets))/(2*eps)
        assert torch.allclose(logits.grad, hand, atol=1e-10)
        assert torch.allclose(logits.grad, numerical, atol=1e-7)
        results = {'seed': SEED, 'ce_hand_gradient_error': float((logits.grad-hand).abs().max()),
                   'ce_finite_difference_error': float((logits.grad-numerical).abs().max())}
        print(results)
        bad_prediction, bad_target = torch.zeros(4, 1), torch.ones(4)
        print('Broadcast bug:', (bad_prediction-bad_target).shape, 'expected [4,1]')
        """), md("## 3. 同一参数初始化，SGD 与 AdamW 的训练轨迹\n只在 train 更新；validation loss 用于诊断。训练预算、weight decay 与学习率不同，结果不能当 optimizer 普遍排名。"), code("""
        import copy
        data_generator = torch.Generator().manual_seed(SEED)
        x = torch.randn(500, 4, generator=data_generator)
        y = (x[:, :1] + .6*x[:, 1:2]**2 + .1*torch.randn(500, 1, generator=data_generator))
        xtr, xva, ytr, yva = x[:350], x[350:], y[:350], y[350:]
        torch.manual_seed(SEED)
        initial = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 1))
        trajectories = {}; final_metrics = {}
        for name in ('SGD', 'AdamW'):
            model = copy.deepcopy(initial)
            optimizer = (torch.optim.SGD(model.parameters(), lr=.04, momentum=.9) if name=='SGD'
                         else torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01))
            trace = []
            for step in range(300):
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss_value = F.mse_loss(model(xtr), ytr)
                loss_value.backward(); optimizer.step()
                if step % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        trace.append([step, float(F.mse_loss(model(xtr), ytr)), float(F.mse_loss(model(xva), yva))])
            trajectories[name] = np.asarray(trace)
            final_metrics[name] = {'train_mse': trace[-1][1], 'validation_mse': trace[-1][2]}
        for name, trace in trajectories.items():
            plt.plot(trace[:, 0], trace[:, 2], label=name)
        plt.xlabel('step'); plt.ylabel('validation MSE'); plt.legend(); plt.show()
        results['optimization'] = final_metrics
        """), md("## 4. train/eval 不等于有无 autograd\nDropout 的 eval 输出确定；BatchNorm 的 running stats 在常见默认配置下仅在 train 更新。"), code("""
        dropout = nn.Dropout(.5); inputs = torch.ones(32, 8)
        dropout.train(); train_a, train_b = dropout(inputs), dropout(inputs)
        dropout.eval(); eval_a, eval_b = dropout(inputs), dropout(inputs)
        assert torch.equal(eval_a, eval_b)
        batch_norm = nn.BatchNorm1d(8)
        batch_norm.train(); batch_norm(inputs * 3)
        running_after_train = batch_norm.running_mean.clone()
        batch_norm.eval(); batch_norm(inputs * 20)
        assert torch.equal(batch_norm.running_mean, running_after_train)
        leaf = torch.tensor(2., requires_grad=True)
        (leaf**2).backward(); first_gradient = float(leaf.grad)
        (leaf**2).backward(); accumulated = float(leaf.grad)
        assert accumulated == 2*first_gradient
        print('Dropout train outputs differ:', not torch.equal(train_a, train_b))
        print('Backward accumulates:', first_gradient, '->', accumulated)
        results['gradient_accumulation'] = [first_gradient, accumulated]
        """), md("## 5. LoRA：B=0 时，首步 A 的零梯度不等于 frozen\nbase weight 冻结，A 随机、B 全零。只检查一步的梯度与参数量，不训练或下载大模型。"), code("""
        class TinyLoRA(nn.Module):
            def __init__(self, in_dim=6, out_dim=4, rank=2, alpha=2):
                super().__init__()
                self.base = nn.Linear(in_dim, out_dim, bias=False)
                self.base.requires_grad_(False)
                self.A = nn.Parameter(.1*torch.randn(rank, in_dim))
                self.B = nn.Parameter(torch.zeros(out_dim, rank))
                self.scale = alpha/rank
            def forward(self, value):
                return self.base(value) + self.scale*(value @ self.A.T @ self.B.T)
        lora = TinyLoRA()
        sample = torch.randn(12, 6); target = torch.randn(12, 4)
        F.mse_loss(lora(sample), target).backward()
        a_grad = float(lora.A.grad.norm()); b_grad = float(lora.B.grad.norm())
        assert lora.base.weight.grad is None and a_grad == 0.0 and b_grad > 0
        results['lora'] = {'trainable_parameters': sum(p.numel() for p in lora.parameters() if p.requires_grad),
                           'base_parameters': lora.base.weight.numel(), 'first_A_grad_norm': a_grad,
                           'first_B_grad_norm': b_grad}
        print(results['lora'])
        """)]
    return finish(cells, "1. 手推 softmax+CE 的 p-y，解释 mean reduction。\n2. 如果忘记 zero_grad，数值会怎样？\n3. 为什么 eval 不阻断 gradient？\n4. LoRA 的两个低秩因子为什么首步梯度不同？")


def ml04():
    cells = start("ML04：PCA、K-means 与低方差信号反例", "区分重建信息、聚类结构与任务相关信息；所有 fitted transforms 只读 train。", sklearn=True)
    cells += [md("## 2. PCA 和 K-means：先 train fit，再 held-out transform\n只用标签评价 ARI，不把标签输入聚类。cluster ID 没有固定 class 含义。"), code("""
        from sklearn.datasets import make_blobs
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, accuracy_score
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import LogisticRegression
        X, labels = make_blobs(n_samples=700, centers=3, n_features=5, cluster_std=2., random_state=SEED)
        X = X * np.array([10., 1., .2, 3., .5])
        permutation = rng.permutation(len(X)); tr, ev = permutation[:500], permutation[500:]
        scaler = StandardScaler().fit(X[tr])
        representations = {'raw': (X[tr], X[ev]),
                           'scaled': (scaler.transform(X[tr]), scaler.transform(X[ev]))}
        rows = []; fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, (name, (train_x, eval_x)) in zip(axes, representations.items()):
            pca = PCA(n_components=2).fit(train_x)
            train_z, eval_z = pca.transform(train_x), pca.transform(eval_x)
            reconstruction = pca.inverse_transform(eval_z)
            cluster = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(train_z)
            predicted = cluster.predict(eval_z)
            rows.append({'representation': name, 'held_out_ARI': adjusted_rand_score(labels[ev], predicted),
                         'held_out_reconstruction_mse_in_this_scale': float(np.mean((eval_x-reconstruction)**2)),
                         'train_explained_variance_fraction': float(pca.explained_variance_ratio_.sum())})
            axis.scatter(eval_z[:, 0], eval_z[:, 1], c=predicted, s=14)
            axis.set(title=name+' PCA / K-means', xlabel='PC1', ylabel='PC2')
        plt.tight_layout(); plt.show(); display(pd.DataFrame(rows))
        results = {'seed': SEED, 'pca_kmeans': rows,
                   'note': 'Raw/scaled reconstruction MSE use different units and are not directly comparable.'}
        """), md("## 3. SVD 写出 PCA 的 projection/reconstruction"), code("""
        train_mean = X[tr].mean(axis=0, keepdims=True)
        centered = X[tr]-train_mean
        U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:2].T
        z = (X[ev]-train_mean) @ components
        recovered = z @ components.T + train_mean
        print('X:', X[ev].shape, 'components:', components.shape, 'Z:', z.shape, 'reconstruction:', recovered.shape)
        plt.plot(np.cumsum(singular_values**2)/np.sum(singular_values**2), marker='o')
        plt.xlabel('component index (zero-based)'); plt.ylabel('cumulative variance fraction'); plt.show()
        """), md("## 4. 低方差信号也可能决定标签\n两个预先固定的 pipeline 对比；不依据 held-out 结果调参。第 1 维大方差但无标签信息，第 2 维小方差却含标签。"), code("""
        low_y = rng.integers(0, 2, size=900)
        low_x = np.column_stack([rng.normal(scale=10., size=900),
                                .05*(2*low_y-1) + rng.normal(scale=.005, size=900)])
        low_tr, low_ev = np.arange(650), np.arange(650, 900)
        baselines = {
            'both standardized features': make_pipeline(StandardScaler(), LogisticRegression()),
            'raw PCA to one component': make_pipeline(PCA(n_components=1), StandardScaler(), LogisticRegression()),
        }
        counterexample = {}
        for name, model in baselines.items():
            model.fit(low_x[low_tr], low_y[low_tr])
            counterexample[name] = float(accuracy_score(low_y[low_ev], model.predict(low_x[low_ev])))
        results['low_variance_signal_counterexample'] = counterexample
        print(counterexample)
        """)]
    return finish(cells, "1. PCA 最大化方差为什么不能保证保留分类信号？\n2. 为什么 cluster IDs 不能直接与 labels 算 accuracy？\n3. raw/scaled reconstruction MSE 能否直接比较？\n4. FP2 reconstruction 很好是否必然说明接触控制有效？")


def ml05():
    cells = start("ML05：cross-attention compressor 与 frozen sidecar", "训练一个合成上下文瓶颈，再冻结它预测 frame/mask/wrench；这是 FP2 思想的教学实例，不是真实 FP2 复现。", torch=True)
    cells += [md("## 2. 先验证 causal mask 的信息依赖\n这里 `nn.MultiheadAttention` 的 bool mask 中 True 表示禁止读取。扰动未来，过去输出应保持不变。"), code("""
        torch.manual_seed(SEED)
        attention = nn.MultiheadAttention(16, 4, dropout=0., batch_first=True).eval()
        sequence = torch.randn(2, 6, 16)
        causal_mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
        changed = sequence.clone(); changed[:, 4:] += 100
        with torch.no_grad():
            original = attention(sequence, sequence, sequence, attn_mask=causal_mask, need_weights=False)[0]
            perturbed = attention(changed, changed, changed, attn_mask=causal_mask, need_weights=False)[0]
        causal_delta = float((original[:, :4]-perturbed[:, :4]).abs().max())
        assert causal_delta < 1e-5
        print('past-output delta after changing future:', causal_delta)
        """), md("## 3. 合成 C 与交互 histories，按样本分 train/val/test\nC 只含 task latent 的特征，不含 W/P。frame/mask/wrench 标签由已知合成规则生成；不使用论文数据或真实 RFM。"), code("""
        generator = torch.Generator().manual_seed(SEED+20)
        count, tokens, input_width, width, horizon = 768, 16, 32, 16, 10
        task_latent = torch.randn(count, 6, generator=generator)
        base = .2*torch.randn(tokens, input_width, generator=generator)
        modes = .2*torch.randn(6, tokens, input_width, generator=generator)
        C = base.unsqueeze(0) + torch.einsum('br,rnd->bnd', task_latent, modes)
        contact = torch.randint(0, 2, (count,), generator=generator).float()
        P = .05*torch.randn(count, horizon, 9, generator=generator)
        P[:, :, 3:] += torch.tensor([1., 0., 0., 0., 1., 0.])
        W = .02*torch.randn(count, horizon, 6, generator=generator)
        W[:, :, 2] += contact[:, None]*(1.+4.*torch.rand(count, 1, generator=generator))
        selection_target = torch.zeros(count, 6); selection_target[:, 2] = contact
        wrench_target = torch.zeros(count, 6)
        wrench_target[:, 2] = contact*(4.+2.*torch.sigmoid(task_latent[:, 0]))
        angle = .5*torch.tanh(task_latent[:, 1])
        c, s = torch.cos(angle), torch.sin(angle)
        first_column = torch.stack([c, torch.zeros_like(c), -s], dim=-1)
        second_column = torch.tensor([0., 1., 0.]).expand(count, -1)
        frame_target = torch.cat([torch.zeros(count, 3), first_column, second_column], dim=-1)
        order = torch.randperm(count, generator=generator)
        train_ids, val_ids, test_ids = order[:512], order[512:640], order[640:]
        print('C', C.shape, '| P', P.shape, '| W', W.shape)
        print('targets:', frame_target.shape, selection_target.shape, wrench_target.shape)
        """), md("## 4. Learned query → cross-attention → 单 token\nEncoder 的 forward 只接收 C。decoder 的 N 个 queries 只读取 z；没有 C→decoder 信息旁路。"), code("""
        class CrossBlock(nn.Module):
            def __init__(self, d=16, heads=4):
                super().__init__()
                self.qnorm = nn.LayerNorm(d); self.knorm = nn.LayerNorm(d)
                self.attn = nn.MultiheadAttention(d, heads, dropout=0., batch_first=True)
                self.norm = nn.LayerNorm(d)
                self.mlp = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
            def forward(self, query, context):
                kv = self.knorm(context)
                query = query + self.attn(self.qnorm(query), kv, kv, need_weights=False)[0]
                return query + self.mlp(self.norm(query))
        class Compressor(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(input_width, width)
                self.query = nn.Parameter(.02*torch.randn(1, 1, width))
                self.blocks = nn.ModuleList([CrossBlock(width), CrossBlock(width)])
            def forward(self, context):
                x = self.projection(context)
                query = self.query.expand(len(context), -1, -1)
                for block in self.blocks:
                    query = block(query, x)
                return query
        class ReconstructionDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.queries = nn.Parameter(.1*torch.randn(1, tokens, width))
                self.block = CrossBlock(width)
                self.output = nn.Linear(width, input_width)
            def forward(self, z):
                queries = self.queries.expand(len(z), -1, -1)
                return self.output(self.block(queries, z))
        compressor, decoder = Compressor(), ReconstructionDecoder()
        with torch.no_grad():
            print('projected:', compressor.projection(C[:2]).shape,
                  '| z:', compressor(C[:2]).shape, '| reconstruction:', decoder(compressor(C[:2])).shape)
        """), md("## 5. 仅用 train C 训练重建\n本toy用MSE+0.1×token-wise cosine distance，没有force loss，未训练真实RFM。2026-11-09源码对照：所读OpenPI/JAX RLT使用MSE；通用PyTorch分支可另配cosine，不能混为同一训练实现。这里保留教学目标。"), code("""
        COMPRESSOR_STEPS = 300
        optimizer = torch.optim.AdamW(list(compressor.parameters())+list(decoder.parameters()), lr=.003)
        reconstruction_trace = []
        for step in range(COMPRESSOR_STEPS):
            batch = train_ids[torch.randint(len(train_ids), (64,))]
            target = C[batch]; reconstructed = decoder(compressor(target))
            objective = F.mse_loss(reconstructed, target) + .1*(1-F.cosine_similarity(reconstructed, target, dim=-1)).mean()
            optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
            if step % 10 == 0:
                with torch.no_grad():
                    val_mse = float(F.mse_loss(decoder(compressor(C[val_ids])), C[val_ids]))
                reconstruction_trace.append([step, float(objective.detach()), val_mse])
        trace = np.asarray(reconstruction_trace)
        plt.plot(trace[:, 0], trace[:, 2]); plt.xlabel('step'); plt.ylabel('validation reconstruction MSE'); plt.show()
        """), md("## 6. 冻结 compressor，训练三头 force sidecar\n本toy选择较小的GRU+concat+MLP，标签为合成数据。2026-11-09所选peg/RLT1024源码另有FiLM→两层128D GRU→FiLM→两路gated fusion、frame9/force6/mask6三头；本cell不实现该完整结构。先用toy理解冻结，再回课程3d对照真实配置。"), code("""
        compressor.eval().requires_grad_(False)
        for parameter in compressor.parameters():
            parameter.grad = None
        with torch.no_grad():
            cached_z = compressor(C)
        class ForceSidecar(nn.Module):
            def __init__(self):
                super().__init__()
                self.history = nn.GRU(15, 32, batch_first=True)
                self.fusion = nn.Sequential(nn.Linear(width+32, 64), nn.GELU())
                self.frame = nn.Linear(64, 9); self.mask = nn.Linear(64, 6); self.wrench = nn.Linear(64, 6)
            def forward(self, z, p_history, w_history):
                _, hidden = self.history(torch.cat([p_history, w_history], dim=-1))
                fused = self.fusion(torch.cat([z.squeeze(1), hidden[-1]], dim=-1))
                return self.frame(fused), self.mask(fused), self.wrench(fused)
        sidecar = ForceSidecar(); optimizer = torch.optim.AdamW(sidecar.parameters(), lr=.003)
        FORCE_STEPS = 300; force_trace = []
        for step in range(FORCE_STEPS):
            batch = train_ids[torch.randint(len(train_ids), (64,))]
            frame, mask_logits, desired = sidecar(cached_z[batch], P[batch], W[batch])
            objective = (F.mse_loss(frame, frame_target[batch]) +
                         F.mse_loss(desired, wrench_target[batch]) +
                         .1*F.binary_cross_entropy_with_logits(mask_logits, selection_target[batch]))
            optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
            if step % 10 == 0:
                force_trace.append([step, float(objective.detach())])
        assert all(p.grad is None for p in compressor.parameters())
        plt.plot(*np.asarray(force_trace).T); plt.xlabel('step'); plt.ylabel('training sidecar loss'); plt.show()
        """), md("## 7. 固定模型后的 held-out 评价与 frame 解码\n报告的是 synthetic labels 上的预测误差，不是力控成功率。6D rotation code 解码为 3×3 正交矩阵；9 个编码数不是 9 个物理自由度。"), code("""
        sidecar.eval()
        with torch.no_grad():
            frame, mask_logits, desired = sidecar(cached_z[test_ids], P[test_ids], W[test_ids])
            mask = (mask_logits.sigmoid() >= .5).float()
            first = F.normalize(frame[:, 3:6], dim=-1)
            second_raw = frame[:, 6:9] - (frame[:, 6:9]*first).sum(-1, keepdim=True)*first
            second = F.normalize(second_raw, dim=-1)
            third = torch.linalg.cross(first, second, dim=-1)
            rotation = torch.stack([first, second, third], dim=-1)
            results = {'seed': SEED, 'causal_perturbation_delta': causal_delta,
                'test_reconstruction_mse': float(F.mse_loss(decoder(cached_z[test_ids]), C[test_ids])),
                'test_frame_code_mse': float(F.mse_loss(frame, frame_target[test_ids])),
                'test_contact_axis_mask_accuracy': float((mask[:, 2]==selection_target[test_ids, 2]).float().mean()),
                'test_desired_wrench_mse': float(F.mse_loss(desired, wrench_target[test_ids])),
                'rotation_shape': list(rotation.shape),
                'compressor_frozen': all(not p.requires_grad for p in compressor.parameters()),
                'scope': 'Synthetic RFM-context surrogate and supervised sidecar. No robot/controller or FP2 performance reproduction.'}
        print(json.dumps(results, indent=2))
        """)]
    return finish(cells, "1. Q 来自哪里，为什么输出只有一个 token？\n2. 哪个阶段允许 force loss 更新 compressor？本实验中答案是没有。\n3. 修改 d，观察 reconstruction 与下游误差是否同步。\n4. 在 validation 上诊断 zero/shuffled context 或 force，解释 OOD 干预的局限。\n5. mask/desired wrench 与实际 controller command 有什么区别？")


def ml06():
    cells = start("ML06：学习 conditional diffusion 与 flow", "在相同合成双峰 action distribution 上训练两个小网络，再从噪声采样；不使用 oracle endpoint 作为推理输入。", torch=True)
    cells += [md("## 2. 同一数据、同一 MLP 结构\n每条样本独立，train/validation/test 隔离。hidden branch 不输入网络；不把采样模式距离叫 robot success。"), code("""
        import copy
        from time import perf_counter
        gen = torch.Generator().manual_seed(SEED+30)
        observations = 2*torch.rand(3600, 1, generator=gen)-1
        branches = 2*torch.randint(0, 2, (3600, 1), generator=gen)-1
        actions = .5*observations + branches + .05*torch.randn(3600, 1, generator=gen)
        train_obs, train_act = observations[:2400], actions[:2400]
        val_obs, val_act = observations[2400:3000], actions[2400:3000]
        test_obs, test_act = observations[3000:], actions[3000:]
        class ConditionalNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
            def forward(self, state, obs, time):
                return self.net(torch.cat([state, obs, time], dim=-1))
        initial = ConditionalNet()
        flow, diffusion = copy.deepcopy(initial), copy.deepcopy(initial)
        TRAIN_STEPS, BATCH_SIZE = 800, 128
        """), md("## 3. Flow：noise→data，target=a−epsilon\n采样时网络只读当前 x、observation 和 tau。固定训练预算，不依据 test 选择 steps。"), code("""
        optimizer = torch.optim.Adam(flow.parameters(), lr=.002)
        flow_trace = []
        for step in range(TRAIN_STEPS):
            ids = torch.randint(len(train_obs), (BATCH_SIZE,))
            obs, endpoint = train_obs[ids], train_act[ids]
            noise = torch.randn_like(endpoint); tau = torch.rand(BATCH_SIZE, 1)
            state = (1-tau)*noise + tau*endpoint
            target = endpoint-noise
            objective = F.mse_loss(flow(state, obs, tau), target)
            optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
            if step % 20 == 0:
                flow_trace.append((step, float(objective.detach())))
        """), md("## 4. DDPM：预测 epsilon\n固定 48-step schedule。训练随机采 k；采样再走完整 reverse chain。两个 loss 目标不同，不横比数值大小。"), code("""
        DIFFUSION_STEPS = 48
        beta = torch.linspace(.001, .2, DIFFUSION_STEPS)
        alpha = 1-beta; alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_previous = torch.cat([torch.ones(1), alpha_bar[:-1]])
        posterior_variance = beta*(1-alpha_bar_previous)/(1-alpha_bar)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=.002)
        diffusion_trace = []
        for step in range(TRAIN_STEPS):
            ids = torch.randint(len(train_obs), (BATCH_SIZE,))
            obs, endpoint = train_obs[ids], train_act[ids]
            k = torch.randint(DIFFUSION_STEPS, (BATCH_SIZE,))
            noise = torch.randn_like(endpoint)
            cumulative = alpha_bar[k, None]
            state = cumulative.sqrt()*endpoint + (1-cumulative).sqrt()*noise
            time = k[:, None].float()/(DIFFUSION_STEPS-1)
            objective = F.mse_loss(diffusion(state, obs, time), noise)
            optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
            if step % 20 == 0:
                diffusion_trace.append((step, float(objective.detach())))
        print('Final forward alpha_bar:', float(alpha_bar[-1]))
        """), md("## 5. 两个 inference sampler：不接受真实 endpoint\nFlow 使用 Euler；DDPM 使用 epsilon prediction 对应的 reverse mean 与 posterior variance。输出每个 observation 的一个 sample。"), code("""
        @torch.no_grad()
        def sample_flow(model, obs, steps=32):
            model.eval(); state = torch.randn(len(obs), 1)
            for index in range(steps):
                tau = torch.full((len(obs), 1), index/steps)
                state = state + model(state, obs, tau)/steps
            return state
        @torch.no_grad()
        def sample_diffusion(model, obs):
            model.eval(); state = torch.randn(len(obs), 1)
            for k in reversed(range(DIFFUSION_STEPS)):
                time = torch.full((len(obs), 1), k/(DIFFUSION_STEPS-1))
                predicted_noise = model(state, obs, time)
                mean = (state-beta[k]/torch.sqrt(1-alpha_bar[k])*predicted_noise)/torch.sqrt(alpha[k])
                state = mean if k == 0 else mean+torch.sqrt(posterior_variance[k])*torch.randn_like(state)
            return state
        """), md("## 6. 固定配置的 held-out 采样指标与 o=0 分布\n记录采样耗时，不把 CPU wall time 推广成真实 VLA latency。曲线不保证一次训练就覆盖所有模式。"), code("""
        metrics = {}; sample_sets = {}
        samplers = {'flow': lambda obs: sample_flow(flow, obs, steps=32),
                    'diffusion': lambda obs: sample_diffusion(diffusion, obs)}
        for name, sampler in samplers.items():
            torch.manual_seed(SEED+100)
            begin = perf_counter(); samples = sampler(test_obs); elapsed = perf_counter()-begin
            centered = samples-.5*test_obs
            distance = torch.minimum((centered-1).abs(), (centered+1).abs())
            metrics[name] = {'test_sample_mode_compatibility': float((distance<.15).float().mean()),
                             'test_positive_mode_fraction': float((centered>0).float().mean()),
                             'sampling_seconds_cpu': elapsed, 'sample_count': len(test_obs)}
            sample_sets[name] = sampler(torch.zeros(2000, 1)).numpy().ravel()
        results = {'seed': SEED, 'train_steps_per_model': TRAIN_STEPS, 'metrics': metrics,
                   'mode_distance_threshold': .15, 'flow_sampling_steps': 32,
                   'diffusion_sampling_steps': DIFFUSION_STEPS,
                   'scope': 'Learned 1D conditional synthetic distributions. No robot rollout or paper reproduction.'}
        display(pd.DataFrame(metrics).T)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        for axis, (name, samples) in zip(axes, sample_sets.items()):
            axis.hist(samples, bins=60, range=(-2, 2), density=True, alpha=.7)
            axis.axvline(-1, linestyle='--', color='black'); axis.axvline(1, linestyle='--', color='black')
            axis.set(title=name, xlabel='sample at o=0', ylabel='density')
        plt.tight_layout(); plt.show()
        """)]
    return finish(cells, "1. 手推 flow target，解释为什么 sampler 不读 expert endpoint。\n2. Diffusion 训练随机抽一步，为什么采样仍是多步？\n3. 在 validation 上尝试不同 Euler steps，固定配置后再查看 test。\n4. 为什么 training MSE、mode coverage 与 closed-loop performance 是不同指标？")


def vla01():
    cells = start("VLA01：RGB/text/state → tokens → action field", "用小型随机 Transformer 追一次完整 VLA tensor、mask、loss 与 sampler；不下载模型，不进行本地/云端预训练复现。", torch=True)
    cells += [md("## 2. 合成 observation/action contract\n2相机、6个text slots、1个state token、8个未来动作。动作是7个合成连续坐标，没有真实机器人控制语义。部分text/action slots无效。"), code("""
        BATCH, CAMERAS, IMAGE_SIZE, PATCH = 2, 2, 32, 8
        TEXT_LENGTH, HORIZON, ACTION_DIM, WIDTH = 6, 8, 7, 32
        images = torch.rand(BATCH, CAMERAS, 3, IMAGE_SIZE, IMAGE_SIZE)
        text_ids = torch.randint(1, 64, (BATCH, TEXT_LENGTH))
        text_valid = torch.ones(BATCH, TEXT_LENGTH, dtype=torch.bool)
        text_valid[1, -2:] = False
        proprio = torch.randn(BATCH, 7)
        coordinate_scale = torch.linspace(.2, 1., ACTION_DIM)
        synthetic_train_actions = torch.randn(24, HORIZON, ACTION_DIM)*coordinate_scale
        train_mean = synthetic_train_actions.mean(dim=(0, 1))
        train_std = synthetic_train_actions.std(dim=(0, 1), correction=0).clamp_min(1e-6)
        clean_coordinates = torch.randn(BATCH, HORIZON, ACTION_DIM)*coordinate_scale
        clean_actions = (clean_coordinates-train_mean)/train_std
        action_valid = torch.ones(BATCH, HORIZON, dtype=torch.bool)
        action_valid[1, -2:] = False
        noise = torch.randn_like(clean_actions)
        tau = torch.rand(BATCH, 1, 1)
        noisy_actions = (1-tau)*noise + tau*clean_actions
        velocity_target = clean_actions-noise
        print('RGB:', images.shape, '| text:', text_ids.shape, '| state:', proprio.shape)
        print('clean/noisy/target:', clean_actions.shape, noisy_actions.shape, velocity_target.shape)
        """), md("## 3. 三条输入支路与 P/S/A block mask\n这是共享小Transformer，不是π0的多expert实现。P读P；S读P/S；A读P/S/A。Noisy A 可双向读取，future sensor不能因此进入条件。"), code("""
        class TinyVLATrace(nn.Module):
            def __init__(self):
                super().__init__()
                patches = (IMAGE_SIZE//PATCH)**2
                self.patch = nn.Conv2d(3, WIDTH, kernel_size=PATCH, stride=PATCH)
                self.camera = nn.Parameter(.02*torch.randn(1, CAMERAS, 1, WIDTH))
                self.patch_position = nn.Parameter(.02*torch.randn(1, 1, patches, WIDTH))
                self.text = nn.Embedding(64, WIDTH)
                self.text_position = nn.Parameter(.02*torch.randn(1, TEXT_LENGTH, WIDTH))
                self.state = nn.Linear(7, WIDTH)
                self.action = nn.Linear(ACTION_DIM, WIDTH)
                self.action_position = nn.Parameter(.02*torch.randn(1, HORIZON, WIDTH))
                self.time = nn.Sequential(nn.Linear(1, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
                layer = nn.TransformerEncoderLayer(WIDTH, 4, dim_feedforward=64, dropout=0., batch_first=True)
                self.backbone = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
                self.output = nn.Linear(WIDTH, ACTION_DIM)
            def forward(self, rgb, ids, text_mask, state, noisy, generation_time):
                b, cameras = rgb.shape[:2]
                patches = self.patch(rgb.flatten(0, 1)).flatten(2).transpose(1, 2)
                patches = patches.reshape(b, cameras, -1, WIDTH)
                image_tokens = (patches+self.camera+self.patch_position).flatten(1, 2)
                text_tokens = self.text(ids)+self.text_position
                prefix = torch.cat([image_tokens, text_tokens], dim=1)
                state_token = self.state(state).unsqueeze(1)
                action_tokens = (self.action(noisy)+self.action_position+
                                 self.time(generation_time.reshape(b, 1)).unsqueeze(1))
                n_prefix = prefix.shape[1]; total = n_prefix+1+HORIZON
                dependency = torch.zeros(total, total, dtype=torch.bool)
                dependency[:n_prefix, n_prefix:] = True
                dependency[n_prefix:n_prefix+1, n_prefix+1:] = True
                key_valid = torch.cat([torch.ones(b, image_tokens.shape[1], dtype=torch.bool), text_mask,
                                       torch.ones(b, 1+HORIZON, dtype=torch.bool)], dim=1)
                hidden = self.backbone(torch.cat([prefix, state_token, action_tokens], dim=1),
                                       mask=dependency, src_key_padding_mask=~key_valid)
                velocity = self.output(hidden[:, -HORIZON:])
                shapes = {'image_tokens': list(image_tokens.shape), 'text_tokens': list(text_tokens.shape),
                          'prefix': list(prefix.shape), 'state_token': list(state_token.shape),
                          'action_tokens': list(action_tokens.shape), 'hidden': list(hidden.shape),
                          'velocity': list(velocity.shape), 'dependency_mask': list(dependency.shape)}
                return velocity, hidden[:, :n_prefix], dependency, shapes
        model = TinyVLATrace()
        predicted, _, dependency, shapes = model(images, text_ids, text_valid, proprio, noisy_actions, tau)
        display(pd.DataFrame({'shape': shapes}).T)
        assert predicted.shape == clean_actions.shape
        plt.imshow((~dependency).numpy(), vmin=0, vmax=1)
        plt.xlabel('key'); plt.ylabel('query'); plt.title('Allowed P/S/A dependencies'); plt.show()
        """), md("## 4. Masked loss 与一次 backward\n不是训练成功记录；只检查模型收到的目标、分母和gradient。Invalid target不计loss，不默认把loss mask用作suffix attention mask。"), code("""
        model.zero_grad(set_to_none=True)
        valid_scalar = action_valid.unsqueeze(-1).expand_as(predicted)
        assert int(valid_scalar.sum()) > 0
        objective = ((predicted-velocity_target).square()*valid_scalar).sum()/valid_scalar.sum()
        objective.backward()
        gradients = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        assert all(math.isfinite(value) for value in gradients.values())
        print('masked flow loss:', float(objective.detach()), '| valid scalars:', int(valid_scalar.sum()))
        print('vision gradient norm:', gradients['patch.weight'])
        """), md("## 5. 多层信息依赖的 perturbation checks\nNo dropout；改变noisy actions不应改prefix。改变无效text padding不应改有效action outputs。此检查说明cache依赖条件，不实现KV cache。"), code("""
        model.eval()
        with torch.no_grad():
            original_v, original_prefix, _, _ = model(images, text_ids, text_valid, proprio, noisy_actions, tau)
            _, changed_prefix, _, _ = model(images, text_ids, text_valid, proprio, noisy_actions+100., tau)
            changed_text = text_ids.clone(); changed_text[~text_valid] = 63
            changed_v, _, _, _ = model(images, changed_text, text_valid, proprio, noisy_actions, tau)
            prefix_delta = float((original_prefix-changed_prefix).abs().max())
            pad_delta = float((original_v-changed_v).abs().max())
        assert prefix_delta < 1e-5 and pad_delta < 1e-5
        print('prefix delta:', prefix_delta, '| text-padding action delta:', pad_delta)
        """), md("## 6. Noise→action sampler 与反归一化trace\n随机网络的输出没有任务能力。Sampler不接收clean expert endpoint；为可读性每步仍重算全部小网络。统计来自独立synthetic train数据。"), code("""
        @torch.no_grad()
        def sample_chunk(model, rgb, ids, valid_text, state, steps=4):
            model.eval(); chunk = torch.randn(len(rgb), HORIZON, ACTION_DIM)
            for index in range(steps):
                time = torch.full((len(rgb), 1, 1), index/steps)
                vector_field, _, _, _ = model(rgb, ids, valid_text, state, chunk, time)
                chunk = chunk+vector_field/steps
            return chunk
        recovered = clean_actions*train_std+train_mean
        assert torch.allclose(recovered, clean_coordinates, atol=1e-6)
        sampled = sample_chunk(model, images, text_ids, text_valid, proprio)
        output_coordinates = sampled*train_std+train_mean
        results = {'seed': SEED, 'shapes': shapes, 'masked_loss_at_random_initialization': float(objective.detach()),
                   'prefix_perturbation_delta': prefix_delta, 'padding_perturbation_delta': pad_delta,
                   'sample_shape': list(output_coordinates.shape), 'optimizer_updates': 0,
                   'scope': 'Random-weight tensor/gradient/mask trace. No pretrained VLA, robot command, or performance validation.'}
        print(json.dumps(results, indent=2))
        """)]
    return finish(cells, "1. 计算prefix为何有38个tokens，总序列为何47。\n2. 为什么suffix未来位置可双向读，但未来真实force不可读？\n3. 原生action、模型action width、hidden width有何区别？\n4. 解释why cache independence is not an implemented KV cache。\n5. 把camera数改为1需要同时改哪些接口？")


def vla02():
    cells = start("VLA02：多速率执行、插值与规则式接触调节", "保持相同1 kHz输出网格，独立配置sensor、context和regulator频率；用简化spring contact解释feedback与stale context。")
    cells += [md("## 2. 声明所有时钟与简化物理假设\n不是FP2模型或Flexiv控制器。x理想跟踪，只有z接触运动；无惯性/摩擦，F=K max(surface−z,0)。低频输出端点之间在1 kHz网格线性插值。"), code("""
        OUTPUT_HZ, SENSOR_HZ, CONTEXT_HZ = 1000, 100, 2
        DURATION, STIFFNESS, NOMINAL_STIFFNESS = 8., 1500., 1200.
        GAIN, MAX_Z_SPEED, SENSOR_DELAY = .0015, .03, .010
        dt = 1./OUTPUT_HZ
        time = np.arange(int(DURATION*OUTPUT_HZ))/OUTPUT_HZ
        x = .02*time
        surface = .004*np.sin(2*np.pi*x/.12) + .001*(time>=5.25)
        true_goal = np.where(time<4.25, 8., 12.)
        sensor_stride = OUTPUT_HZ//SENSOR_HZ
        context_stride = OUTPUT_HZ//CONTEXT_HZ
        sensor_noise = np.random.default_rng(SEED).normal(scale=.05, size=len(time))
        def goal_at(t):
            return 8. if t<4.25 else 12.
        # Shared nominal RFM-like motion reference: task goal updated at 2 Hz,
        # geometric waypoints at 50 Hz, same interpolated reference for all methods.
        reference_time = np.arange(int(DURATION*50)+1)/50.
        context_sample_time = np.floor(reference_time*CONTEXT_HZ)/CONTEXT_HZ
        reference_goal = np.where(context_sample_time<4.25, 8., 12.)
        reference_z = .002*np.sin(2*np.pi*.02*reference_time/.12)-reference_goal/NOMINAL_STIFFNESS
        interpolated_motion = np.interp(time, reference_time, reference_z)
        """), md("## 3. 同一个plant与sensor；四种执行策略\nPosition-only使用共享motion reference。其余在z轴使用规则式导纳，x继续相同运动。Slow/fast使用相同增益以隔离频率，不宣称它们分别经过最优调参。Stale-context仅让下游目标力保持8 N。"), code("""
        def rollout(policy_hz=50, position_only=False, stale_context=False):
            assert OUTPUT_HZ % policy_hz == 0
            policy_stride = OUTPUT_HZ//policy_hz
            z = float(interpolated_motion[0])
            start_z = end_z = z; last_policy_step = 0
            measured = 0.; cached_goal = goal_at(0.)
            force_log = np.zeros(len(time)); z_log = np.zeros(len(time))
            available_samples = np.full(len(time), np.nan)
            age_log = np.full(len(time), np.nan)
            policy_updates = sensor_updates = context_updates = 0
            for index, timestamp in enumerate(time):
                actual_force = STIFFNESS*max(surface[index]-z, 0.)
                force_log[index] = actual_force; z_log[index] = z
                if index % sensor_stride == 0:
                    available_samples[index] = actual_force+sensor_noise[index]
                    sensor_updates += 1
                source_index = int(np.floor((timestamp-SENSOR_DELAY)*SENSOR_HZ+1e-9))*sensor_stride
                if source_index >= 0:
                    assert source_index <= index and np.isfinite(available_samples[source_index])
                    measured = float(available_samples[source_index])
                    age_log[index] = timestamp-source_index/OUTPUT_HZ
                if index % context_stride == 0:
                    if not stale_context:
                        cached_goal = goal_at(timestamp)
                    context_updates += 1
                if position_only:
                    z = float(interpolated_motion[min(index+1, len(time)-1)])
                else:
                    if index % policy_stride == 0:
                        velocity = float(np.clip(-GAIN*(cached_goal-measured), -MAX_Z_SPEED, MAX_Z_SPEED))
                        start_z, end_z = z, z+velocity/policy_hz
                        last_policy_step = index; policy_updates += 1
                    fraction = min((index-last_policy_step+1)/policy_stride, 1.)
                    z = (1-fraction)*start_z+fraction*end_z
            return {'force': force_log, 'z': z_log, 'age': age_log,
                    'policy_updates': policy_updates, 'sensor_updates': sensor_updates,
                    'context_updates': context_updates}
        traces = {
            'position only': rollout(position_only=True),
            '2 Hz force regulator': rollout(policy_hz=2),
            '50 Hz force regulator': rollout(policy_hz=50),
            '50 Hz, stale force context': rollout(policy_hz=50, stale_context=True),
        }
        """), md("## 4. 评价力误差、contact loss与新信息次数\n统一排除最初0.5s启动段；sensor在该段之前无历史测量，以0初始化。这里的1 kHz只表示共同输出网格，不表示1 kHz新sensor/learned predictions。"), code("""
        evaluation = time>=.5; rows = []
        for name, trace in traces.items():
            error = trace['force'][evaluation]-true_goal[evaluation]
            rows.append({'method': name, 'force_MAE_N': float(np.abs(error).mean()),
                         'force_peak_N': float(trace['force'].max()),
                         'contact_loss_fraction': float((trace['force'][evaluation]<.5).mean()),
                         'rule_updates': trace['policy_updates'], 'new_sensor_samples': trace['sensor_updates'],
                         'context_checks': trace['context_updates'], 'output_samples': len(time)})
        display(pd.DataFrame(rows))
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(time, true_goal, '--', color='black', label='desired force')
        for name, trace in traces.items():
            axes[0].plot(time, trace['force'], label=name, alpha=.8)
        axes[0].set(ylabel='force (N)', title='Shared output grid, different feedback/context updates')
        axes[0].legend()
        axes[1].plot(time, surface, '--', color='black', label='surface')
        for name, trace in traces.items():
            axes[1].plot(time, trace['z'], label=name, alpha=.8)
        axes[1].set(xlabel='time (s)', ylabel='z (m)'); plt.tight_layout(); plt.show()
        results = {'seed': SEED, 'output_hz': OUTPUT_HZ, 'sensor_hz': SENSOR_HZ,
                   'context_hz': CONTEXT_HZ, 'sensor_delay_seconds': SENSOR_DELAY,
                   'excluded_startup_seconds': .5, 'metrics': rows,
                   'scope': 'Rule-based regulator and simplified spring contact; no learned FP2 or robot-performance validation.'}
        """)]
    return finish(cells, "1. 输出8000个点是否意味着8000次策略决策？\n2. 把sensor降到25 Hz，为什么提高输出插值率不能消除sensor age？\n3. 为什么stale context可能给出稳定但不正确的目标力？\n4. 同增益频率对比与各自最优调参比较有何不同？\n5. 加入惯性、摩擦或旋转后，当前实验哪些结论需重新验证？")


def main():
    builders = {
        "Lab01_probabilistic_bc.ipynb": original_bc,
        "Lab02_episode_windows.ipynb": original_windows,
        "Pi_mechanisms.ipynb": original_pi,
        "ML01_linear_logistic.ipynb": ml01,
        "ML02_models_and_evaluation.ipynb": ml02,
        "ML03_autograd_optimization.ipynb": ml03,
        "ML04_pca_kmeans.ipynb": ml04,
        "ML05_attention_compressor.ipynb": ml05,
        "ML06_diffusion_flow.ipynb": ml06,
        "VLA01_token_action_trace.ipynb": vla01,
        "VLA02_multirate_force_execution.ipynb": vla02,
    }
    for name, builder in builders.items():
        save(name, builder())
        print("Built", name)
    manifest = {"builder": "../tools/build_colab_notebooks.py", "execution": "not executed; empty outputs",
                "notebooks": {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in builders},
                "embedded_script_sources": {name: hashlib.sha256((LABS/name).read_bytes()).hexdigest()
                    for name in ("01_probabilistic_bc.py", "02_episode_windows.py", "pi_series_mechanics.py")}}
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
