```



git fetch upstream
git merge upstream/main

先备份并确保工作区干净
git status
git branch backup/main-before-upstream-20260302
添加上游仓库（只需一次）
git remote add upstream https://github.com/TheSmallHanCat/flow2api.git
git remote -v

拉取上游最新代码
git fetch upstream

同步到你的 main（推荐用 merge，简单稳妥）
git checkout main
git pull --ff-only origin main
git merge upstream/main


待办：多账号同时自动化登录时可能有问题，
需要考虑多账号同时登录时，如何处理账号切换的问题，比如如何判断当前登录的账号，如何切换账号等。


Linux 部署 


sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv

cd /path/to/flow2api
cp config/setting_example.toml config/setting.toml

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

# 默认配置是 browser 打码，首次建议安装 Chromium
python -m playwright install chromium

python main.py


