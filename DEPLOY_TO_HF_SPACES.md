# 部署到 Hugging Face Spaces 指南

本指南将帮助您将 SHAPE 项目部署到 Hugging Face Spaces。

## 📋 前置要求

1. **Hugging Face 账户**
   - 访问 https://huggingface.co/ 注册账户
   - 获取访问令牌：https://huggingface.co/settings/tokens

2. **Git**
   - 确保已安装 Git
   - 配置 Git 用户信息

3. **Hugging Face CLI**（可选但推荐）
   ```bash
   pip install huggingface_hub[cli]
   ```

## 🚀 部署步骤

### 方法一：使用 Git（推荐）

#### 1. 初始化 Git 仓库（如果还没有）

```bash
cd "D:\1-Data_Analysis\Code\HF clone\SHAPE"
git init
git add .
git commit -m "Initial commit: SHAPE application"
```

#### 2. 在 Hugging Face 创建 Space

1. 访问 https://huggingface.co/spaces
2. 点击 "Create new Space"
3. 填写信息：
   - **Space name**: `your-username/shape` (例如: `username/shape`)
   - **SDK**: 选择 `Gradio`
   - **Hardware**: 根据需求选择（CPU 或 GPU）
   - **Visibility**: Public 或 Private
4. 点击 "Create Space"

#### 3. 添加远程仓库并推送

```bash
# 添加 Hugging Face Space 作为远程仓库
git remote add origin https://huggingface.co/spaces/your-username/shape

# 或者使用 SSH（如果已配置）
# git remote add origin git@hf.co:spaces/your-username/shape

# 推送代码
git push origin main
```

如果遇到分支名称问题（可能是 `master` 而不是 `main`）：
```bash
git branch -M main
git push -u origin main
```

### 方法二：使用 Hugging Face CLI

#### 1. 登录 Hugging Face

```bash
huggingface-cli login
# 输入您的访问令牌
```

#### 2. 创建并推送 Space

```bash
# 创建 Space
huggingface-cli repo create shape --type space --sdk gradio

# 克隆 Space 仓库
git clone https://huggingface.co/spaces/your-username/shape
cd shape

# 复制项目文件
cp -r ../SHAPE/* .
cp ../SHAPE/.gitignore .

# 提交并推送
git add .
git commit -m "Initial commit: SHAPE application"
git push
```

## 🔐 配置环境变量（Secrets）

在 Hugging Face Space 中设置环境变量：

1. 进入您的 Space 页面
2. 点击 "Settings" 标签
3. 在 "Repository secrets" 部分添加：
   - **OPENAI_API_KEY**: 您的 OpenAI API 密钥
   - **HUGGINGFACE_TOKEN** (可选): 如果需要访问私有模型

## 📝 文件结构要求

确保以下文件存在于项目根目录：

```
SHAPE/
├── app.py                 # 主应用文件（必需）
├── requirements.txt       # Python 依赖（必需）
├── README.md             # Space 描述（必需）
└── .gitignore            # Git 忽略文件（推荐）
```

## ✅ 验证部署

1. 访问您的 Space URL: `https://huggingface.co/spaces/your-username/shape`
2. 等待构建完成（通常需要 5-10 分钟）
3. 检查日志：
   - 点击 Space 页面的 "Logs" 标签
   - 查看是否有错误信息

## 🔧 常见问题

### 问题 1: 构建失败

**解决方案**:
- 检查 `requirements.txt` 中的依赖版本是否兼容
- 查看构建日志中的错误信息
- 确保所有必需的 Python 包都已列出

### 问题 2: 运行时错误

**解决方案**:
- 检查环境变量是否正确设置
- 查看应用日志
- 确保 `OPENAI_API_KEY` 已正确配置

### 问题 3: 内存不足

**解决方案**:
- 升级到 GPU 硬件（在 Space Settings 中）
- 优化代码以减少内存使用
- 使用更小的模型或批处理大小

### 问题 4: 端口问题

**解决方案**:
- 确保 `app.py` 中使用端口 7860（Spaces 标准端口）
- 检查 `IS_SPACES` 环境变量检测是否正确

## 📦 更新部署

当您需要更新代码时：

```bash
# 修改代码后
git add .
git commit -m "Update: description of changes"
git push origin main
```

Spaces 会自动检测更改并重新构建。

## 🎯 最佳实践

1. **测试本地运行**: 在推送前确保应用在本地正常运行
2. **使用 .gitignore**: 避免推送不必要的文件（缓存、临时文件等）
3. **环境变量**: 永远不要将 API 密钥提交到代码中，使用 Secrets
4. **依赖管理**: 固定依赖版本以避免兼容性问题
5. **日志监控**: 定期检查 Space 日志以发现潜在问题

## 📚 相关资源

- [Hugging Face Spaces 文档](https://huggingface.co/docs/hub/spaces)
- [Gradio 文档](https://gradio.app/docs/)
- [Git 基础教程](https://git-scm.com/book)

## 🆘 获取帮助

如果遇到问题：
1. 查看 Hugging Face Spaces 文档
2. 检查 Space 的构建日志
3. 在 Hugging Face 论坛寻求帮助

