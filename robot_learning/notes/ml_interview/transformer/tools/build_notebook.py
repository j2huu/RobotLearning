"""Build TR01 using only the Python standard library; does not execute cells."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.strip().splitlines(keepends=True)})


def code(source):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': source.strip().splitlines(keepends=True),
                  'execution_count': None, 'outputs': []})


md('''# TR01：Attention、Mask 与 KV cache 的最小机制实验
对应 T1/T3/T4/T6/T8。只依赖 Python 标准库、固定小输入，无安装、下载或仓库路径。

先预测结果，再逐 cell 运行。运行到最后应打印 8 个机制检查通过。
本 notebook 不训练模型：两层 toy 只有 identity Q/K/V attention + residual，没有 FFN、norm、learned projection。
它检查固定输入上的数学与信息依赖，不证明 PyTorch kernel、Colab 平台或机器人性能。

各 cell 输出保持空白；执行记录见讲义 VALIDATION.md，学习者需另记自己的结果。''')
code('''import math
from copy import deepcopy

checks = []

def close(a, b, tol=1e-10):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return abs(a - b) <= tol

def attention(q, k, v, allowed):
    # 无 batch/head 轴：[Nq,d] x [Nk,d] -> [Nq,dv]
    out = []
    for qi, row in zip(q, allowed):
        if not any(row):
            raise ValueError('No legal key for this query')
        scores = [sum(a*b for a, b in zip(qi, kj))/math.sqrt(len(qi)) for kj in k]
        peak = max(s for s, ok in zip(scores, row) if ok)
        weights = [math.exp(s-peak) if ok else 0.0 for s, ok in zip(scores, row)]
        denom = sum(weights)
        out.append([sum(w*vj[d] for w, vj in zip(weights, v))/denom for d in range(len(v[0]))])
    return out

def causal(q_pos, k_pos):
    return [[j <= i for j in k_pos] for i in q_pos]

def add(a, b):
    return [[x+y for x, y in zip(ar, br)] for ar, br in zip(a, b)]

def block(x, allowed):
    return add(x, attention(x, x, x, allowed))

def stack(x, allowed, layers=2):
    x = deepcopy(x)
    for _ in range(layers):
        x = block(x, allowed)
    return x
''')
md('''## 1. 手算 softmax：先回答 8 和 2 从哪里来
Q/K 的宽度为 1，直接让 logits 成为 [0, ln(3)]。固定 Q/K，只改允许集合。''')
code('''q, k, v = [[1.0]], [[0.0], [math.log(3)]], [[2.0], [10.0]]
assert close(attention(q, k, v, [[True, True]]), [[8.0]])
assert close(attention(q, k, v, [[True, False]]), [[2.0]])
checks.append('weighted average and mask')
print('允许两项 -> 8；只允许第一项 -> 2')''')
md('''## 2. 未来扰动：检验两层，而不只看三角矩阵
只改最后一个 token，前三个位置应保持不变。然后故意使用全可见矩阵，确认这个输入确实能够揭示泄漏。''')
code('''x = [[0.1, 0.3], [0.4, -0.2], [-0.1, 0.5], [0.2, 0.1]]
x_future = deepcopy(x)
x_future[-1] = [1.5, -0.7]
mask = causal(range(4), range(4))
a, b = stack(x, mask), stack(x_future, mask)
assert close(a[:3], b[:3])
assert not close(a[-1], b[-1])
all_visible = [[True]*4 for _ in range(4)]
assert not close(stack(x, all_visible)[0], stack(x_future, all_visible)[0])
checks.append('two-layer causal perturbation with negative control')
print('causal：过去输出不变；全可见对照：过去输出改变')''')
md('''## 3. Padding：只屏蔽 key，不等于删除 query
这里最后两项是假 padding。改变它们的内容，前两个有效位置的输出不应改变。每层都应用相同 key 有效性限制。''')
code('''key_valid = [True, True, False, False]
pad_mask = [key_valid[:] for _ in range(4)]
x_pad = deepcopy(x)
x_pad[2:] = [[9.0, 8.0], [-7.0, 6.0]]
assert close(stack(x, pad_mask)[:2], stack(x_pad, pad_mask)[:2])
assert not close(stack(x, all_visible)[:2], stack(x_pad, all_visible)[:2])
checks.append('padding isolation')
print('有效输出不受padding内容影响；无效query自身不保证为零')''')
md('''## 4. 间接泄漏：C 不直接读 D，仍然可能泄漏
顺序为 C、P、D。C 可以读 P，P 可以读 D。先预测一层与两层的区别。
修复方式是同时禁止 P 读取 D，切断整条间接路径。''')
code('''z = [[0.2, 0.1], [0.3, -0.2], [0.4, 0.5]]
z_changed = deepcopy(z)
z_changed[2] = [1.8, -0.9]
leaky = [[True, True, False], [False, True, True], [False, False, True]]
assert close(stack(z, leaky, 1)[0], stack(z_changed, leaky, 1)[0])
assert not close(stack(z, leaky, 2)[0], stack(z_changed, leaky, 2)[0])
safe = [[True, True, False], [False, True, False], [False, False, True]]
assert close(stack(z, safe, 2)[0], stack(z_changed, safe, 2)[0])
checks.append('indirect leakage and repair')
print('一层C不变；两层C泄漏；切断P读取D后恢复隔离')''')
md('''## 5. 空行必须有明确策略
参考实现拒绝全屏蔽 query，而不是依赖 NaN 或给所有位置同一个有限负数。生产实现可对无效 query 设计安全占位规则。''')
code('''try:
    attention([[1.0]], [[1.0]], [[2.0]], [[False]])
except ValueError as exc:
    print('预期错误：', exc)
else:
    raise AssertionError('Empty mask was not rejected')
checks.append('empty-row rejection')''')
md('''## 6. RoPE：相对位移的二维检查
固定内容 Q/K，比较同时平移位置前后的点积。这里只演示一个二维频率，不代表完整位置泛化能力。''')
code('''def rotate(vec, pos, theta=math.pi/2):
    c, s = math.cos(pos*theta), math.sin(pos*theta)
    return [c*vec[0]-s*vec[1], s*vec[0]+c*vec[1]]

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

base_q, base_k = [1.0, 0.0], [1.0, 0.0]
assert close(dot(rotate(base_q, 0), rotate(base_k, 1)), 0.0)
assert close(dot(rotate(base_q, 1), rotate(base_k, 2)), 0.0)
assert close(dot(rotate(base_q, 2), rotate(base_k, 2)), 1.0)
checks.append('RoPE relative-position identity')
print('位移1：内积0；位移0：内积1')''')
md('''## 7. 两层 KV cache：与 full causal forward 比较
每层保存自己的输入 K/V。投影设为 identity；这让读者直接检查依赖和缓存位置，不需权重训练。
旧层 cache 不能用最后一层输出代替。''')
code('''def cached_stack(seq, layers=2):
    caches = [[] for _ in range(layers)]
    outputs = []
    for token in seq:
        h = token[:]
        for layer in range(layers):
            caches[layer].append(h[:])
            # 新query处在已存在keys的末尾，所有keys都合法。
            mixed = attention([h], caches[layer], caches[layer], [[True]*len(caches[layer])])[0]
            h = [a+b for a, b in zip(h, mixed)]
        outputs.append(h)
    return outputs

full = stack(x, causal(range(len(x)), range(len(x))), 2)
cached = cached_stack(x, 2)
assert close(full, cached)
max_error = max(abs(a-b) for ar, br in zip(full, cached) for a, b in zip(ar, br))
checks.append('two-layer full versus incremental cache')
print('full/cache最大绝对差:', max_error)''')
md('''## 8. 非方阵 mask：错位也会导致不等价
过去3项，新query在全局位置3和4。检查正确允许矩阵，并与错误的局部左上三角作对照。
注意：这里是单层非方阵读取测试；上一节的两层cache测试没有加入RoPE。''')
code('''keys = x + [[0.7, -0.4]]
queries = keys[3:]
global_mask = causal([3, 4], range(5))
local_wrong = causal(range(2), range(5))
assert global_mask == [[True, True, True, True, False], [True]*5]
good = attention(queries, keys, keys, global_mask)
full_last = attention(keys, keys, keys, causal(range(5), range(5)))[3:]
wrong = attention(queries, keys, keys, local_wrong)
assert close(good, full_last)
assert not close(good, wrong)
checks.append('non-square global-position mask')
print('正确mask:', global_mask)
print('错误mask:', local_wrong)
assert len(checks) == 8
print('PASS:', len(checks), 'mechanism checks')
for name in checks:
    print(' -', name)''')
md('''## 运行后自己回答
1. 两层泄漏实验为何一层正常、两层失败？
2. 为什么只有一个新query时不应盲目使用局部1×N下三角？
3. Padding query为什么不自动为零？
4. 如果把identity投影换成learned Linear，正确性测试应保留哪些？
5. 如果在cache实验中加入RoPE，应该在哪里记录和使用逻辑位置？

记录Python版本、8项检查结果和自己修改的反例。这里没有训练、GPU计时、任务成功率或真实机器人实验。''')

for i, cell in enumerate(cells):
    cell['id'] = f'tr01-{i:02d}'

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = ROOT / 'notebooks' / 'TR01_attention_masks_and_cache.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
print(out.relative_to(ROOT))
