# 部署到现有 Space: 5xuekun/SHAPE

本指南将帮助您将代码推送到已存在的 Hugging Face Space。

## 🚀 快速部署

### 方法一：使用 PowerShell 脚本（推荐）

```powershell
.\deploy_to_existing_space.ps1
```

脚本会自动：
1. 初始化 Git（如果需要）
2. 配置远程仓库指向 `5xuekun/SHAPE`
3. 提交所有更改
4. 推送到 Space（会替换现有内容）

### 方法二：手动 Git 命令

```bash
# 1. 初始化 Git（如果还没有）
git init

# 2. 添加所有文件
git add .

# 3. 提交更改
git commit -m "Update: SHAPE application with analysis visualizer"

# 4. 添加/更新远程仓库
git remote add origin https://huggingface.co/spaces/5xuekun/SHAPE
# 或者如果已存在，更新它：
# git remote set-url origin https://huggingface.co/spaces/5xuekun/SHAPE

# 5. 推送到 Space（强制推送以替换现有内容）
git push -u origin main --force
```

如果您的默认分支是 `master`：
```bash
git branch -M main
git push -u origin main --force
```

## ⚠️ 重要提示

1. **强制推送**: 使用 `--force` 会替换 Space 中的所有现有内容
2. **备份**: 如果 Space 中有重要内容，建议先备份
3. **认证**: 确保您有该 Space 的写入权限

## 🔐 验证环境变量

部署后，确保在 Space Settings 中配置了：

1. 访问：https://huggingface.co/spaces/5xuekun/SHAPE/settings
2. 在 "Repository secrets" 中检查：
   - **OPENAI_API_KEY**: 必须设置
   - **HUGGINGFACE_TOKEN** (可选): 如果需要

## ✅ 验证部署

1. 访问 Space: https://huggingface.co/spaces/5xuekun/SHAPE
2. 等待构建完成（查看 Logs 标签）
3. 测试应用功能

## 📝 更新代码

以后更新时：
```bash
git add .
git commit -m "Update: description"
git push origin main --force
```

## 🆘 常见问题

### 问题 1: 权限错误
**解决方案**: 
- 确保您是 Space 的所有者或有写入权限
- 检查 Hugging Face 账户权限

### 问题 2: 推送被拒绝
**解决方案**:
- 使用 `--force` 标志强制推送
- 确保已登录：`huggingface-cli login`

### 问题 3: 构建失败
**解决方案**:
- 检查 `requirements.txt` 中的依赖
- 查看 Space 的构建日志
- 确保 `app.py` 文件存在且正确

