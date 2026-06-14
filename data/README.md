# Data Directory

此目录只放小型元数据、切分清单、统计结果和开发用 manifest。

大型数据集、原始图像、缓存和中间产物应放在：

```text
/data1/liuxuan/datasets
```

项目输出、日志和 checkpoint 应放在：

```text
/data1/liuxuan/logs
```

或项目本地已忽略的 `outputs/` 目录。

v0 manifest 建议放在：

```text
outputs/manifests/
```

不要把原始图片、预测缓存、提交 ZIP 或训练输出提交进仓库。
