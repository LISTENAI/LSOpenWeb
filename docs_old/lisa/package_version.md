---
id: package_version
title: LISA 项目的版本控制
slug: /package_version
---

<p style={{color: "#6d6d6d", fontSize: "24px"}}>本文介绍 LISA 项目的软件包版本管理机制。</p>

## 关于 NPM

LPM（聆思软件包管理平台） 是基于 NPM 搭建的私有软件包管理平台，遵循 NPM 的规范与约定。LISA 命令行工具也支持使用众多 NPM 软件包。你可以在[这里](http://nodejs.cn/learn)对 NPM 进行更深入的了解。


## 依赖声明

所有 LISA 项目都存在 `package.json` 文件，在此文件中的 `dependencies` 字段设置了作为依赖安装的软件包列表。以 CSK 语音交互项目中 `package.json` 的为例，其中声明了固件源码、算法、版型模板、项目生成器、LISA 核心库等软件包。

```shell
  "dependencies": {
    "@algo/csk4002-cae-mlp": "^1.0.1",
    "@board/lskits-csk4002": "^1.0.2",
    "@generator/csk": "^1.1.5",
    "@listenai/lisa_core": "^1.1.0",
    "@source/csk4002": "~3.1.4"
  }
```


## 版本规则

语义版本控制的概念很简单：所有的版本都有 3 个数字：`x.y.z`。


- 第一个数字是主版本。
- 第二个数字是次版本。
- 第三个数字是补丁版本。


当发布新的版本时，不仅仅是随心所欲地增加数字，还要遵循以下规则：


- 当进行不兼容的更改时，则升级主版本。
- 当以向后兼容的方式添加功能时，则升级次版本。
- 当进行向后兼容的缺陷修复时，则升级补丁版本。


每个 LISA 软件包都必须遵守该约定，这一点非常重要，因为整个系统都依赖于此。

为什么这么重要？

因为以下一些规则，可用于在 `package.json` 文件中选择要将软件包更新到的版本（当运行 `lisa upgrade` 时）。

规则使用了这些符号：

- **^**
-  **~**
-  **>**
-  **>=**
- **<**
- **<=**
- **=**
- **-**
- **||**

这些规则的详情如下：

- **^:** 只会执行不更改最左边非零数字的更新。 如果写入的是 ^0.13.0，则当运行 npm update 时，可以更新到 0.13.1、0.13.2 等，但不能更新到 0.14.0 或更高版本。 如果写入的是 ^1.13.0，则当运行 `lisa upgrade` 时，可以更新到 1.13.1、1.14.0 等，但不能更新到 2.0.0 或更高版本。
- **~:** 如果写入的是 〜0.13.0，则当运行 `lisa upgrade` 时，会更新到补丁版本：即 0.13.1 可以，但 0.14.0 不可以。
- **>:** 接受高于指定版本的任何版本。
- **>=:** 接受等于或高于指定版本的任何版本。
- **<=:** 接受等于或低于指定版本的任何版本。
- **<:** 接受低于指定版本的任何版本。
- **=:** 接受确切的版本。
- **-:** 接受一定范围的版本。例如：2.1.0 - 2.6.2。
- **||:** 组合集合。例如 < 2.1 || > 2.6。

可以合并其中的一些符号，例如 `1.0.0 || >=1.1.0 <1.2.0`，即使用 1.0.0 或从 1.1.0 开始但低于 1.2.0 的版本。

还有其他的规则：

无符号: 仅接受指定的特定版本（例如 `1.2.1`）。
`latest`: 使用可用的最新版本。
