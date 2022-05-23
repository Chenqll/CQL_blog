# build oneflow
- https://github.com/Oneflow-Inc/OneTeam/blob/63083c576fd1f6c627832651fad1ab1813028105/tutorial/howto_build_oneflow.md
- 补充
[远程下载 miniconda](https://blog.csdn.net/weixin_43264420/article/details/118179287)
- 删除build环境重新编译
  rm -f *
- 在15，16号机器，编译需要在gcc环境下，将clang环境改为gcc    
```
conda env create -f=dev/clang10/environment-v2.yml--->conda env create -f=dev/gcc7/environment-v2.yml
```
- 删除前面的build目录 然后新建一个build 重新cmake

- 缺少libunwind包，需要conda安装，需要在conda环境激活之前install，找conda包的途径 https://anaconda.org/conda-forge/libunwind

- 得出结论在15号机器别用ninja，成功的参考issue https://github.com/Oneflow-Inc/conda-env 主要就是用make -j4 的命令代替ninja -j4

- 注意：编译成功的oneflow每次都需要先进入conda环境：然后检查是否有编译好的环境：
```
(base) [chenqiaoling@oneflow-15 ~]$ conda activate oneflow-dev-gcc7-v2
(oneflow-dev-gcc7-v2) [chenqiaoling@oneflow-15 ~]

source build/source.sh

(oneflow-dev-gcc7-v2) [chenqiaoling@oneflow-15 ~]$ python -m oneflow --doctor
path: ['/home/chenqiaoling/oneflow/python/oneflow']
version: 0.8.0+cu102.git.2083cc9
git_commit: 2083cc9
cmake_build_type: Release
rdma: False
mlir: False
```
# 一些遇到的接口
- einsum 爱因斯坦求和

    爱因斯坦简记法：是一种由爱因斯坦提出的，对向量、矩阵、张量的求和运算 $\sum$ 的求和简记法。

    在该简记法当中，省略掉的部分是：1）求和符号 $\sum$ 与2）求和号的下标 i 

    省略规则为：默认成对出现的下标（如下例1中的i和例2中的k）为求和下标。

    比如用 $x_i$ , $y_i$ 简化表示内积<X,Y>

    ```python
    print(a_tensor)
 
    tensor([[11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
            [41, 42, 43, 44]])
 
    print(b_tensor)
    
    tensor([[1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4]])
    
    # 'ik, kj -> ij'语义解释如下：
    # 输入a_tensor: 2维数组，下标为ik,
    # 输入b_tensor: 2维数组，下标为kj,
    # 输出output：2维数组，下标为ij。
    # 隐含语义：输入a,b下标中相同的k，是求和的下标，对应上面的例子2的公式
    output = torch.einsum('ik, kj -> ij', a_tensor, b_tensor)
    ```
- sampler 采样器

    https://blog.csdn.net/aiwanghuan5017/article/details/102147825
  
    首先需要知道的是所有的采样器都继承自Sampler这个类，如下：

    可以看到主要有三种方法：分别是：

    __init__: 这个很好理解，就是初始化
    __iter__: 这个是用来产生迭代索引值的，也就是指定每个step需要读取哪些数据
    __len__: 这个是用来返回每次迭代器的长度

    **SequentialSampler** 按顺序对数据集采样。其原理是首先在初始化的时候拿到数据集data_source，之后在__iter__方法中首先得到一个和data_source一样长度的range可迭代器。每次只会返回一个索引值。

# docs
- https://github.com/Oneflow-Inc/OneTeam/blob/master/tutorial/howto_write_api_docs.md
- https://github.com/Oneflow-Inc/community/pull/12

```
  git clone git@github.com:Oneflow-Inc/oneflow-api-cn.git
  cd oneflow-api-cn
  python3 -m pip install pybind11 --user // pybind11 是一个 header-only 的库，换句话说，只需要 C++ 项目里直接 include pybind11 的头文件就能使用
  python3 -m pip install -r docs/requirements.txt --user
  python3 setup.py install --user
```
- use sphinx
https://zhuanlan.zhihu.com/p/27544821

对整个仓库熟悉

    /oneflow-api-cn/docs/source/cn/activation_ops.py 里的中文是哪来的
        
        首先搞懂英文的是哪来的，是动态设置docstring来的，C++的接口需要添加docstring，于是在 'oneflow/python/oneflow/framework/docstr'下设置.py文件对由C++导出的接口设置docstring  https://github.com/Oneflow-Inc/OneTeam/blob/master/tutorial/howto_write_api_docs.md

        通过调用 docreset.reset_docstr，把原有的 __doc__ 替换为中文翻译。__doc__ 是python对象的一个属性

    /oneflow-api-cn/docs/source/autograd.rst 这个rst文件是什么意思，什么作用

        文本写作文件 https://www.jianshu.com/p/1885d5570b37 内置'..xxxx::' 语法为文本替换 

        rst 文件中导出接口 https://github.com/Oneflow-Inc/OneTeam/blob/master/tutorial/howto_write_api_docs.md

        rst 只是目录文件？？？-配置文档结构？？


        rst导出C++接口变成python对象，.py接收这个对象并修改内置__doc__属性

    
     make html 命令，其实就是利用了工具 sphinx-build，将 oneflow 中对象的 docstring 提取出来，生成 HTML 文件。

     make html 时的内部原理如下，即 sphinx 先读取 conf.py，并在其中 import oneflow，然后读取 *.rst 文件，确认要生成哪些 Python 接口的文档，然后提取它们的 docstring，并生成 HTML 文件。

     docreset/__init__.py 删除的global是什么意思

     https://www.bilibili.com/read/cv11923872


**rst文件与.py文件的关系**

- rst 文件中导出C++接口，.py文件接收每个接口然后调用 docreset.reset_docstr，把原有的 __doc__ 替换为中文翻译，（rst导出C++接口变成python对象，.py接收这个对象并修改内置__doc__属性）。
  
- 这里的中文翻译是人工加上的么？
  
**添加.readthedocs.yaml文件的意义?我了解到的是直接在官网链接仓库 同时配置样式就能直接用rtd。.readthedocs.yaml文件里面的内容我明白是官网上的示例**
- `html_theme = 'sphinx_rtd_theme'`
  
**docreset/__init__.py 里删除的东西的含义FLAG_BUILD 与配置.readthedocs.yaml或者将api.cn加入oneflow主仓库有什么关系**
- 因为要把中文文档加入到oneflow主仓库之中。

- 出问题有可能是oneflow仓库没有更新
  到这里https://docs.oneflow.org/master/index.html下载安装最新的oneflow：
  python3 -m pip install -f https://staging.oneflow.info/branch/master/cu112 --pre oneflow
  出现没有CUDA的问题 也有可能是安装了oneflow的cpu版本
  
- 如果修改了docreset的init文件
  pip uninstall docreset
  python setup.py install 重新安装
  
- doc里的代码test过不了可能是给出的输出和源输出不一致，此时
  ipython 进入python环境
  把 docstring里的打码复制进去，看看是哪里报错

- 5.10使用oneflow-api-cn里面的guide的readme出现的问题
  sphinx-build not found，原因是他不在环境变量里
  首先找到sphinx-build在哪里，发现在~/.local/bin
  ```
  查看一下：
  ~/.local/bin/sphinx-build 
  ```
  于是把它加在环境变量里
  ```bash
  export PATH=~/.local/bin:$PATH 
  ```
- 找不到oneflow 
    ```bash
    (base) [chenqiaoling@oneflow-15 docs]$ make html
    `Running Sphinx v3.5.4

    Configuration error:
    There is a programmable error in your configuration file:

    Traceback (most recent call last):
    ...
    ...
    ModuleNotFoundError: No module named 'oneflow._oneflow_internal'

    make: *** [html] Error 2`
    ```
此时打印一下
```
Successfully installed oneflow-0.8.0.dev20220510+cu102
(base) [chenqiaoling@oneflow-15 ~]$ python -m oneflow --doctor
Traceback (most recent call last):
  File "/home/chenqiaoling/miniconda3/lib/python3.9/runpy.py", line 188, in _run_module_as_main
    mod_name, mod_spec, code = _get_module_details(mod_name, _Error)
  File "/home/chenqiaoling/miniconda3/lib/python3.9/runpy.py", line 147, in _get_module_details
    return _get_module_details(pkg_main_name, error)
  File "/home/chenqiaoling/miniconda3/lib/python3.9/runpy.py", line 111, in _get_module_details
    __import__(pkg_name)
  File "/home/chenqiaoling/oneflow/python/oneflow/__init__.py", line 21, in <module>
    import oneflow._oneflow_internal
ModuleNotFoundError: No module named 'oneflow._oneflow_internal'
```
发现她是在oneflow包里找的，在的 PYTHONPATH 里，导入的第一优先级，是导入你的这个源码里的 oneflow，但是你的源码 oneflow 没有编译通过，所以就凉了。
  
把环境变量删除 ```unset PYTHONPATH```,把这个环境变量删除。就不会第一导入你的源码路径。这样就可以顺位找其它的 oneflow了（pip安装的）

- 关于signature
  如果不加 signature，又是 C++ 直接导出的接口，那么 signature 其实是无参的。就可能是错的。

  如果是 python 接口，不加 signature 是没问题的，会提取正确。

# 如何为 oneflow 写算子
- 实现步骤
- 定义 op --> 