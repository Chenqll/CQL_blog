# docker
https://www.csdn.net/tags/Mtzakg5sMjM3MjUtYmxvZwO0O0OO0O0O.html

什么是docker：docker是虚拟化容器技术，有三个主要概念：镜像（类）、容器（对象）、仓库。docker就是类似VM虚拟机一样的虚拟技术，体积小，运行速度快。

docker作用：可以把代码和环境一起打包部署到生产环境中。比如：我们写好的代码迁移到别的地方运行，不需要重新配置环境就能直接运行

docker底层：是用Go语言编写的

docker的三大特征：镜像、容器、仓库
镜像（类似于一个类）：包含了各种环境或者服务(tomcat)一个模板

容器（对象）：是镜像(run)起来之后的一个实例，可以把容器看做是一个简易版的Linux环境容器就是集装箱(logo上的集装箱)

仓库：是存放镜像的场所，最大的公开库是Docker Hub（https://hub.docker.com/）

# git基本使用和注意事项
- 为了过CI，必须在每一个git clone之后设置用户名和邮箱
  ```bash
  git config --global user.email "gitHub邮箱"
  git config --global user.name "gitHub用户名"
  ···
- 在对主仓库提交PR之前一定要先创建分支，然后在分支上进行你的修改，不然在master主分支上进行的修改是提交不上去的（premission的问题），此时再切换到其他其他分支进行提交是不会有在master上的修改。

- 但是出现上述情况不要慌可以将master上的修改放入缓存，然后切换到别的分支再加载缓存，具体如下：
  ```
  on branch master
  >>> git stash
  >>> git stash list # 打印一下stash
  >>> git stash show -p stash@{0} # 打印一下stash的内容
  >>> git checkout *branch*
  >>> git stash pop # 将stash的缓存加载进来
  ```
- 如果想要找到某个版本和里面的内容使用
  ```
  >>> git reflog #查看历史记录
    fe66181 HEAD@{20}: checkout: moving from cql5.11 to master
    c658c46 HEAD@{21}: checkout: moving from master to cql5.11
    fe66181 HEAD@{22}: checkout: moving from cql5.0 to master
    c658c46 HEAD@{23}: checkout: moving from cql5.11 to cql5.0
    c658c46 HEAD@{24}: checkout: moving from master to cql5.11
    fe66181 HEAD@{25}: commit: 5.11
    59b72cc HEAD@{26}: commit: 5.11
    545367f HEAD@{27}: checkout: moving from cql5.0 to master
    c658c46 HEAD@{28}: checkout: moving from cql5.11 to cql5.0
    c658c46 HEAD@{29}: checkout: moving from cql5.0 to cql5.11
    c658c46 HEAD@{30}: commit: fix
    2bceec6 HEAD@{31}: checkout: moving from cql5.0 to cql5.0
    2bceec6 HEAD@{32}: checkout: moving from master to cql5.0
    545367f HEAD@{33}: checkout: moving from cql5.0 to master
    2bceec6 HEAD@{34}: checkout: moving from master to cql5.0
    545367f HEAD@{35}: commit: fix
    2bceec6 HEAD@{36}: pull origin cql5.0: Fast-forward
    cae837f HEAD@{37}: clone: from git@github.com:Oneflow-Inc/oneflow-api-cn.git
  >>> git reset cae837f # 退回到某个特定的版本
  >>> git status 查看一下是不是自己要的版本
  >>> git reset 2bceec6 # 如果不是就继续遍历所有HEAD
  ```

- 做完上述操作之后很容易产生conflict 于是：


- git log --stat 查看当前工作区将要push的内容

# mkdocs 的使用
- 查看有无 mkdocs，无则安装
  ```
  mkdocs
  ```
- 新建一个项目，在新建的文件夹下：
  ```
  mkdocs new `name`
  ```
- 将 md 文件转为 html
  ```
  cd `name`
  mkdocs build
  ```
# 利用 github pages
- 先将整个 `name` 目录提交到新仓库的master分支
- 创建 gh-pages 分支，将 site 下的 html 文件 放在 gh-pages 的根目录
  ```
  cp -R site/* ./
  ```
- 提交 gh-pages 分支