# Robbyant

用于管理 RoboTwin 与 LingBot-VLA 2.0 的本地开发工作区。顶层仓库保存项目组织、部署脚本和依赖记录；两个上游项目作为 Git 子模块，分别保留自己的代码与提交历史。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `RoboTwin/` | [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 仿真与机器人任务项目 |
| `lingbot-vla-v2/` | [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) 模型与训练项目 |
| `deployment/` | 本地环境安装、验证脚本，以及依赖版本记录 |

子模块固定在本地初始化时使用的版本。上游项目的许可证分别见各自目录。

## 下载完整工作区

```bash
git clone --recurse-submodules https://github.com/HBK-08/Robbyant.git
cd Robbyant
```

如果已经 clone 了顶层仓库，补齐子模块：

```bash
git submodule update --init --recursive
```

项目安装与运行说明见 `RoboTwin/README.md` 和 `lingbot-vla-v2/README.md`。`deployment/` 中的脚本保留当前机器的路径、CUDA、Conda 和显卡配置；换机器时需要按实际环境调整。环境安装包、缓存、模型权重、数据集和运行日志不提交到顶层仓库。

2026 蚂蚁灵波 VLA 挑战赛的训练、评测及提交安排见 [比赛执行计划](docs/competition-plan.md)。训练前准备脚本和独立配置保存在 `deployment/competition-pretrain-20261014/`；数据与权重的实际就绪状态以该目录中的 `pretrain-report.json` 为准。准备脚本不提供训练启动模式。

## 日常提交

顶层文件的修改可以直接提交：

```bash
git add README.md .gitignore deployment/
git commit -m "Update workspace configuration"
git push
```

修改子模块代码时，先在对应子模块内提交，并将提交推送到你有写入权限的仓库；如需使用自己的 fork，应同时更新 `.gitmodules` 中的 URL。然后在顶层提交子模块的新版本引用。这样其他人才能下载到对应代码。
