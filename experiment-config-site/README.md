# 实验配置浏览器

这个本地网站用于读取 `configs/experiments` 下的实验配置，并以卡片、详情页和过程顺序图展示实验信息。
同时会读取 `/mnt/project/world_model/tool_generalist/artifacts` 下的结果 manifest，并按 `source_paths.config` 归并到对应配置详情中。

## 启动

```bash
cd experiment-config-site
npm start
```

默认地址是 `http://127.0.0.1:4173`。可以用环境变量修改端口：

```bash
PORT=4300 npm start
```

## 实现说明

- 后端使用 Node.js 原生 HTTP 服务，不依赖 npm 包。
- Node 后端调用 `src/read_experiments.py` 将 Python dataclass 配置展开为 JSON。
- API：
  - `GET /api/experiments` 返回实验列表摘要。
  - `GET /api/experiments/:id` 返回单个实验的完整配置、步骤参数、源码赋值语句和相关 artifacts 结果。
