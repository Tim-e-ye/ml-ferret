# AutoDL Pro 极简无状态架构与自动化开发全流程手册

本方案专为 **AutoDL Pro 无状态算力容器** + **AutoDL-FS 区域网络存储** + **本地 IDE (VSCode / Antigravity IDE) 自动化直连** + **GitHub 纯代码单向同步** 深度定制。

---

## 目录
1. [核心存储与目录架构](#一核心存储与目录架构极简无状态化)
2. [自动化开机与 IDE 极客连接](#二自动化开机与-ide-极客连接)
3. [镜像与环境生命周期管理（SOP）](#三镜像与环境生命周期管理sop)
4. [训练 I/O 与容灾最佳实践](#四训练-io-与容灾最佳实践)
5. [Git 单向流动纪律与 .gitignore 保护](#五git-单向流动纪律)

---

## 一、核心存储与目录架构（极简无状态化）

Pro 实例没有数据盘，仅有系统盘。为了实现**随用随开、关机即走、安全持久**，目录规范如下：

| 目录层级 | 物理挂载位置 | 典型读写速度 | 承载内容与职责 | 生命周期 |
| :--- | :--- | :--- | :--- | :--- |
| **系统盘 (扩容至 60GB)** | `/root/` 或 `/` | 极高 (本地 NVMe) | **纯代码与执行环境**：Python/Conda、CUDA/PyTorch 库、通过 Git clone 的项目代码仓库。 | 随实例释放重置（通过“自定义镜像”实现环境固化）。 |
| **文件存储 (AutoDL-FS)** | `/root/autodl-fs/` | 高 (同机房内网) | **核心资产仓库**：原始与打包数据集（如 `datasets/*.tar`）、预训练底座权重、微调输出的 Checkpoint。 | **永久持久化**（按容量单独计费，不随实例关机/释放销毁）。 |
| **内存虚拟盘 (RAM Disk)** | `/dev/shm/` | 极致 (数十 GB/s 内存) | **训练中实时解压的数据集**：消除千万张小图片网络寻址延迟，榨干 GPU 算力。 | 随关机清空，训练前从 FS 自动解压导入。 |

> ⚠️ **地域同区强约束**：  
> AutoDL-FS 只能挂载同地域的容器。例如 FS 开在 **“北京 A 区”**，开机 Pro 实例时也必须选择 **“北京 A 区”** 节点。

---

## 二、自动化开机与 IDE 极客连接

我们在本地提供了零依赖的自动化运维脚本 [tools/autodl_pro_manager.py](file:///d:/Code/VibeCode/Antigravity/Project/Ferret/tools/autodl_pro_manager.py)。

### 1. 配置本地密钥与配置文件
复制 [tools/autodl_config.example.json](file:///d:/Code/VibeCode/Antigravity/Project/Ferret/tools/autodl_config.example.json) 为 `tools/autodl_config.json`（该文件已加入 `.gitignore`，不会泄漏到 GitHub）：

```json
{
  "api_token": "填入你的AutoDL控制台API_Token",
  "instance_uuid": "选填，若为空则自动匹配名带pro的实例",
  "ssh_host_alias": "autodl-pro",
  "identity_file": "~/.ssh/id_rsa"
}
```

### 2. 常用操作指令（本地终端执行）

#### 🚀 一键开机并注入 SSH
```bash
python tools/autodl_pro_manager.py start
```
**脚本执行过程：**
1. 自动调用 API 请求开机（`power_on`）。
2. 轮询等待实例转为 `running` 状态。
3. 提取动态分配的公网 SSH 节点与端口号（如 `connect.westb.seetacloud.com:25314`）。
4. 自动修改本地 `~/.ssh/config`，无缝覆写 `Host autodl-pro` 的 `HostName` 与 `Port`。

#### 💻 IDE 秒级连入
- 在 VSCode / Antigravity IDE 远程连接插件中，直接点击连接 **`autodl-pro`**。
- 无需每次开机打开网页控制台复制端口，无需手动修改 config，即刻直连！

#### ⏸️ 一键安全关机（停止 GPU 计费）
```bash
python tools/autodl_pro_manager.py stop
```

#### 📊 状态检查
```bash
python tools/autodl_pro_manager.py status
```

---

## 三、镜像与环境生命周期管理（SOP）

### 阶段 1：基座构建（初次仅需做一次）
1. 在 AutoDL 控制台手动开一台 Pro 实例，将**系统盘扩容至 60GB**，挂载同区 AutoDL-FS。
2. **开启学术网络加速**（由于访问 GitHub / HuggingFace 网络原因，**每个新终端窗口必须先执行**）：
   ```bash
   source /etc/network_turbo
   ```
   > 📌 **重要提醒**：`source /etc/network_turbo` 仅对当前终端窗口有效。每次重新连接 SSH 或新建终端窗口时，若需拉取 GitHub/HuggingFace 资源，必须重新执行一次。

3. 连入实例，配置好 Python 3.10 / PyTorch / CUDA 环境：
   ```bash
   conda activate base # 或自定义 conda env
   pip install --upgrade pip
   pip install -e .
   pip install pycocotools protobuf==3.20.0 ninja opencv-python
   ```
4. 在实例内配置拉取 GitHub 私有库的 SSH Key（可选）：
   ```bash
   ssh-keygen -t ed25519 -C "autodl-ferret"
   cat ~/.ssh/id_ed25519.pub # 将公钥添加到 GitHub -> Settings -> SSH Keys
   ```
4. 将当前实例在控制台**保存为“自定义镜像”**（例如命名为 `ferret-env-v1`）。

### 阶段 2：日常无状态极速开发
此后，开机直接加载该自定义镜像：
- 环境、Python 依赖全在系统盘就绪；
- 连入后仅需：
  ```bash
  cd /root/Ferret
  git pull origin main
  ```
- 即可开始代码编写或模型调试。

### 阶段 3：依赖更新（按需）
如果项目中新增了重度 C++/CUDA 依赖（如 FlashAttention / DeepSpeed 特殊编译版本）：
- `pip install xxx` 或编译完成后，在控制台重新保存镜像覆盖旧版本。

---

## 四、训练 I/O 与容灾最佳实践

### 1. 数据集连续打包存储 (AutoDL-FS)
禁止将零散几十万张小图片散落存在 `/root/autodl-fs/`，网络文件系统的并发寻址开销会导致 GPU 严重掉点（Volatile GPU-Util 忽高忽低）。

**推荐操作：**
在 FS 中存放打包文件：
```bash
# 在 FS 的 datasets 目录存放：
/root/autodl-fs/datasets/go_dataset.tar
```

### 2. 训练前内存盘解压 (`/dev/shm`)
运行我们封装好的 [scripts/autodl_train_env.sh](file:///d:/Code/VibeCode/Antigravity/Project/Ferret/scripts/autodl_train_env.sh)：
- 自动将 `go_dataset.tar` 提取至 `/dev/shm/go_dataset/`；
- DataLoader 从内存直接读取图像张量，速度达 20GB/s 以上，GPU 利用率拉满至 100%。

### 3. Checkpoint 直写持久化 FS
训练参数中：
```bash
--output_dir /root/autodl-fs/models/ferret_go_checkpoint
```
权重生成后直接落地在 AutoDL-FS 内网盘，即使容器实例意外关闭、被抢占或主动关机，权重安然无恙。

---

## 五、Git 单向流动纪律

- **本地开发**：写代码、调逻辑、生成并校验微调 JSON 数据集格式。
- **Push 到 GitHub**：确保只推送纯 Python/Shell/MD 代码（大文件已被 `.gitignore` 过滤，严禁推送 `.pth`、`.bin`、`images/`）。
- **Remote-SSH 实例端**：`git pull` 同步最新逻辑后，在后台通过 `nohup` 或 `tmux` 运行训练脚本。
