
#!/bin/bash

echo "🚀 开始同步代码（正确目录结构）..."

# 配置参数
SERVER="refrain@10.16.45.46"
REMOTE_BASE="/mnt/lyh/DA-FasterCNN/DA-Faster-RCNN"

echo "📦 服务器打包中..."
ssh refrain@10.16.45.46 "cd /mnt/lyh/DA-FasterCNN/DA-Faster-RCNN && tar czf /tmp/code_sync.tar.gz \
    --exclude='datasets' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pth' \
    --exclude='*.model' \
    --exclude='*.jpg' \
    --exclude='*.png' \
    --exclude='DA-Faster-RCNN.png' \
    --exclude='.git'\
    ."

# 下载
echo "⬇️  下载代码包..."
scp refrain@10.16.45.46:/tmp/code_sync.tar.gz ./

# 解压
echo "📂 解压代码..."
tar xzf code_sync.tar.gz --overwrite --exclude='.git'

# 清理
echo "🧹 清理临时文件..."
rm code_sync.tar.gz
ssh refrain@10.16.45.46 "rm /tmp/code_sync.tar.gz"

echo "✅ 同步完成!"
echo ""
echo "📁 同步的文件:"
find . -maxdepth 2 -type f -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.txt" -o -name "*.ipynb" | sort

echo ""
git remote -v

echo ""
git status --short
