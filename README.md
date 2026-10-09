# Robot Learning / VLA 论文与课程

本目录由 Git 管理论文库与 Robot Learning 学习项目。学习入口是 [robot_learning/README](robot_learning/README.md)。

```text
Paper/
├── *.pdf                    # 论文原文
├── robot_learning/
│   ├── README.md            # 学习入口与课序
│   ├── notes/               # 课程讲义、专题笔记、面试题
│   ├── labs/                # 实验代码与说明
│   │   └── results/         # 小型实验结果与报告
│   ├── planning/            # 课程计划与学习进度
│   └── resources/           # 论文索引、主题路线、外部资源
├── .gitignore
└── .gitattributes
```

- [课程目录](robot_learning/README.md)
- [第 4 课：概率视角的 BC](robot_learning/notes/04_probabilistic_bc_and_training.md)
- [第 5 课：训练、数据与观测](robot_learning/notes/05_training_data_and_observation.md)
- [后续课程计划](robot_learning/planning/course_plan.md)
- [论文索引](robot_learning/resources/paper_index.md)

## Git 使用

在本目录执行：

```bash
git status
git diff
git add robot_learning
git commit -m "Study: record lesson notes and experiments"
```

新论文使用 `git add <论文文件名.pdf>` 加入版本管理。论文、讲义、代码及小型结果文件会记录在 Git 中；`tmp/`、Obsidian 本地状态、Python 缓存、虚拟环境和大型训练 checkpoint 由 `.gitignore` 排除。

学习资料有变动时，同时检查课程目录、实验命令与相对链接，并在提交中写清本次学习或实验主题。
