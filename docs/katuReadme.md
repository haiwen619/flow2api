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

