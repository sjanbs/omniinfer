# proxy+vllm mock形态运行指南

测试用例运行前会启动vllm mock形态的后台进程和proxy进程，用例主要是测试端到端的请求发送和响应行为，发送request到proxy，proxy转发到vllm，vllm中模型执行过程被mock了，直接返回随机输出。

## 1. 前置操作

执行命令：

```shell
bash setup_vllm_mock.sh
```

## 2. 快速执行

执行命令：

```shell
pytest test_proxy.py -s -v
```

## 3. 获取覆盖率报告

执行命令：

```shell
bash gen_proxy_cov.sh
```

生成文件存放在当前目录下的```proxy_report```中

## 4. 用例调试

test_proxy.py里面会通过fixture在用例执行前启动vllm_mock进程和proxy进程，调试时通过以下步骤快速调试单个用例：

1. 可以先手动通过脚本```run_vllm_mock.py```和```run_proxy.py```启动

2. 设置环境变量```export SKIP_FIXTURE=1```跳过每次执行pytest的setup步骤

3. 执行单个用例：```pytest test_proxy.py::test_xxx -s -v```
