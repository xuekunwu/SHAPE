# 快速部署指南

## 🚀 一键部署到 Hugging Face Spaces

### 前置步骤

1. **安装 Hugging Face CLI**（如果还没有）
   ```bash
   pip install huggingface_hub[cli]
   ```

2. **登录 Hugging Face**
   ```bash
   huggingface-cli login
   ```
   输入您的访问令牌（在 https://huggingface.co/settings/tokens 获取）

3. **创建 Space**
   - 访问 https://huggingface.co/spaces
   - 点击 "Create new Space"
   - Space name: `your-username/shape`
   - SDK: Gradio
   - 点击 "Create Space"

### 方法一：使用 PowerShell 脚本（Windows）

```powershell
.\deploy.ps1 -SpaceName "your-username/shape"
```

### 方法二：手动 Git 命令

```bash
# 1. 初始化 Git（如果还没有）
git init
git add .
git commit -m "Initial commit: SHAPE application"

# 2. 添加远程仓库
git remote add origin https://huggingface.co/spaces/your-username/shape

# 3. 推送代码
git push -u origin main
```

如果您的默认分支是 `master`：
```bash
git branch -M main
git push -u origin main
```

### 方法三：使用 Hugging Face CLI

```bash
# 创建 Space（如果还没有）
huggingface-cli repo create shape --type space --sdk gradio

# 克隆并推送
git clone https://huggingface.co/spaces/your-username/shape
cd shape
# 复制文件后
git add .
git commit -m "Initial commit"
git push
```

## 🔐 配置 Secrets

部署后，在 Space Settings 中添加：

1. 访问：`https://huggingface.co/spaces/your-username/shape/settings`
2. 在 "Repository secrets" 部分添加：
   - **OPENAI_API_KEY**: 您的 OpenAI API 密钥

## ✅ 验证

1. 访问您的 Space: `https://huggingface.co/spaces/your-username/shape`
2. 等待构建完成（查看 Logs 标签）
3. 测试应用功能

## 📝 更新代码

```bash
git add .
git commit -m "Update: description"
git push origin main
```

## 🆘 常见问题

- **构建失败**: 检查 `requirements.txt` 和构建日志
- **运行时错误**: 检查环境变量和日志
- **认证问题**: 确保已运行 `huggingface-cli login`

详细说明请查看 `DEPLOY_TO_HF_SPACES.md`

