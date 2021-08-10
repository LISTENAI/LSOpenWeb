# 扫描笔 Turnkey 方案

为帮助开发者快速落地自己的扫描笔产品，聆思推出 Turnkey（一站式）解决方案；方案包括整机（笔头模组+上位机）软硬件开发，云端能力对接、整机性能自动化测试与产测指导；从全链路解决客户在开发时遇到的疑难问题，将方案落地时间由 **6个月** 缩短至 **3个月**；



## 方案框架说明 

![](./files/frame.png)

上图描述了整个扫描笔的交互链路，从中可了解整个链路中各环节所起作用：

- 按压笔头开始扫描后，摄像头模组开始工作并将图片以 **66fps**（每秒传66张图）的形式传给 CSK 芯片，CSK 的作用是将传来的图片进行**裁重**，并从每次扫描的第二张图开始，计算出每张图的 **横向与纵向偏移量**；之后将裁重后的图片与偏移量信息通过 **SPI** 传给上位机；
- 上位机通过 SPI 收到 CSK 所传图片与偏移量信息后，将图片 **每5帧** 进行拼接后，通过 **EVS协议** 传至 **iFLYOS**；
- iFLYOS 根据上位机图片与偏移量信息，进行**在线拼接**，将图片传输至 **iFLYOS OCR 识别服务**；
- OCR 识别服务返回识别结果，此时 iFLYOS 再依次调用**分词**（仅支持中文）、**翻译 **与 **TTS合成** 服务；
- iFLYOS 将 **原文识别结果**、**译文结果**，**原文TTS**、**译文TTS** 返回至上位机；
- 上位机进行屏幕展示与TTS播报，本次交互结束；

**简而言之，除上位机固件底层驱动开发与界面UI设计需要开发者自行实现，其他开发项Turnkey方案均已涵盖；**项目落地时间将大幅降低；接下来将详细描述各环节中 Turnkey 提供的产物，以及开发者的使用方法；



## 笔头模组

笔头模组包括摄像头、CMOS、CSK；扫描时的图片采集，裁重都需要依赖笔头模组。是整个扫描笔开发中最重要且复杂的一环。

### 硬件参考设计

针对没有扫描笔开发经验的开发者，Turnkey 提供了标准笔头模组参考设计与核心元器件选型，对笔头开口宽度、摄像头夹角、摄像头距水平面高度等核心数据进行规范，开发者无需再花费时间调试，直接复用即可；

| 文件               | 更新时间   | 操作                                                         |
| ------------------ | ---------- | ------------------------------------------------------------ |
| 笔头原理图参考设计 | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E7%A1%AC%E4%BB%B6%2F%E7%AC%94%E5%A4%B4%E5%8E%9F%E7%90%86%E5%9B%BE%E8%AE%BE%E8%AE%A1%2FBF30A2%EF%BC%88%E7%AC%94%E5%A4%B4%E5%8E%9F%E7%90%86%EF%BC%89.pdf) |

以下是核心元器件参考选型手册：

| 文件            | 更新时间   | 操作                                                         |
| --------------- | ---------- | ------------------------------------------------------------ |
| 摄像头模组      | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E7%A1%AC%E4%BB%B6%2F%E7%AC%94%E5%A4%B4%E5%8E%9F%E7%90%86%E5%9B%BE%E8%AE%BE%E8%AE%A1%2FBF30A2%EF%BC%88%E7%AC%94%E5%A4%B4%E5%8E%9F%E7%90%86%EF%BC%89.pdf) |
| BF30A2（CMOS）  | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E7%A1%AC%E4%BB%B6%2F%E6%A0%B8%E5%BF%83%E5%85%83%E5%99%A8%E4%BB%B6%E8%A7%84%E6%A0%BC%E5%8F%82%E8%80%83%2FSPVS310100.pdf) |
| 笔头按压开关-左 | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E7%A1%AC%E4%BB%B6%2F%E6%A0%B8%E5%BF%83%E5%85%83%E5%99%A8%E4%BB%B6%E8%A7%84%E6%A0%BC%E5%8F%82%E8%80%83%2FSPVS410100.pdf) |
| 笔头按压开关-右 | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E7%A1%AC%E4%BB%B6%2F%E6%A0%B8%E5%BF%83%E5%85%83%E5%99%A8%E4%BB%B6%E8%A7%84%E6%A0%BC%E5%8F%82%E8%80%83%2FSPVS410100.pdf) |

### CSK相关

Turnkey 方案提供集成了裁剪与产测校准算法的 CSK 固件，其中校准算法用于产测时对抠图偏移的情况进行校准；此外 Turnkey 方案还提供了 CSK 与上位机通讯的标准 SPI 与 UART 协议；

| 文件                   | 更新时间   | 操作                                                         |
| ---------------------- | ---------- | ------------------------------------------------------------ |
| 扫描笔CSK固件（0.1.6） | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2FCSK%2F%E6%89%AB%E6%8F%8F%E7%AC%94CSK%E5%9B%BA%E4%BB%B6%EF%BC%880.1.6%EF%BC%89.zip) |
| 扫描笔上位机通信协议   | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2FCSK%2F%E6%89%AB%E6%8F%8F%E7%AC%94%E4%B8%8A%E4%BD%8D%E6%9C%BA%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AEv1.6.pdf) |

### 

## 上位机

### 1.集成上位机SDK

为了帮助开发者快速接入 iFLYOS，Turnkey 方案提供了基于 [LISA API](https://open.listenai.com/resource/open/doc_resource%2F%E8%BD%AF%E4%BB%B6%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97%2FLISA%20API%20%E5%8F%82%E8%80%83%E6%89%8B%E5%86%8C.pdf) 实现的应用层 SDK， 功能包括：

- 可直连 iFLYOS，无需另外实现鉴权逻辑；
- 可使用云端的 OCR、翻译、词典、语音交互等能力；
- 提供支持产测的接口，如录音质量检测；

| 文件            | 更新时间   | 操作                                                         |
| --------------- | ---------- | ------------------------------------------------------------ |
| EVS_SDK_release | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E4%B8%8A%E4%BD%8D%E6%9C%BA%2FEVS_SDK-20210730-release.7z) |

### 2.不集成SDK，自行开发上层应用

若开发者不便集成 SDK，Turnkey 方案提供了云端接口，帮助开发者打通云端链路；

| 文件             | 说明                                                    | 更新时间   | 操作                                                         |
| ---------------- | ------------------------------------------------------- | ---------- | ------------------------------------------------------------ |
| 隐式授权接口     | 通过隐式授权的方式，使设备通过iFLYOS云端鉴权            | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E6%95%B4%E6%9C%BA%E6%B5%8B%E8%AF%95%2F%E6%89%AB%E6%8F%8F%E7%AC%94%E8%87%AA%E5%8A%A8%E5%8C%96%E6%B5%8B%E8%AF%95%E6%96%B9%E6%A1%88v1.0.2.pdf) |
| 扫描笔webAPI接口 | 包括请求OCR、口语评测、翻译的标准协议，以及词典接口调用 | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E4%B8%8A%E4%BD%8D%E6%9C%BA%2F%E6%89%AB%E6%8F%8F%E7%AC%94webAPI%E6%8E%A5%E5%8F%A3%20-%20iFLYOS.pdf) |



此外，无论是否集成SDK，为保证更快的交互响应速度和更好的用户体验，我们建议在开发上位机固件时，参考以下文档：

| 文件                     | 更新时间   | 操作                                                         |
| ------------------------ | ---------- | ------------------------------------------------------------ |
| 扫描笔上位机开发优化建议 | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E4%B8%8A%E4%BD%8D%E6%9C%BA%2F%E6%89%AB%E6%8F%8F%E7%AC%94%E4%B8%8A%E4%BD%8D%E6%9C%BA%E5%BC%80%E5%8F%91%E4%BC%98%E5%8C%96%E5%BB%BA%E8%AE%AE.pdf) |



## 整机测试

整机开发完成后，无论是正式出货前，还是后续每次固件更新迭代，都需要进行整机性能测试，确保固件可用性；为此，聆思根据扫描笔手持滑动的特性，以**机械臂**为工具载体，输出一整套整机加接口的软件自动化测试和硬件自动化方案，⼤幅提升测试效率，加快发版速度。

| 文件                 | 更新时间   | 操作                                                         |
| -------------------- | ---------- | ------------------------------------------------------------ |
| 扫描笔自动化测试方案 | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E6%95%B4%E6%9C%BA%E6%B5%8B%E8%AF%95%2F%E6%89%AB%E6%8F%8F%E7%AC%94%E8%87%AA%E5%8A%A8%E5%8C%96%E6%B5%8B%E8%AF%95%E6%96%B9%E6%A1%88v1.0.2.pdf) |



## 产测方案

根据开发者产品需求与实际产线情况，聆思分别制定了精简版与完整版两套产测方案，建议对产测要求较高的开发者采用完整版方案；

| 文件                                | 更新时间   | 操作                                                         |
| ----------------------------------- | ---------- | ------------------------------------------------------------ |
| 扫描笔产线生产测试流程指导 - 精简版 | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E4%BA%A7%E6%B5%8B%2F%E6%89%AB%E6%8F%8F%E7%AC%94%E4%BA%A7%E7%BA%BF%E7%94%9F%E4%BA%A7%E6%B5%8B%E8%AF%95%E6%B5%81%E7%A8%8B%E6%8C%87%E5%AF%BC%20-%20%E7%B2%BE%E7%AE%80%E7%89%88.pdf) |
| 扫描笔产线生产测试流程指导 - 完整版 | 2021.08.09 | [下载](https://open.listenai.com/resource/open/doc_resource%2F%E6%89%AB%E6%8F%8F%E7%AC%94%2F%E4%BA%A7%E6%B5%8B%2F%E6%89%AB%E6%8F%8F%E7%AC%94%E4%BA%A7%E7%BA%BF%E7%94%9F%E4%BA%A7%E6%B5%8B%E8%AF%95%E6%B5%81%E7%A8%8B%E6%8C%87%E5%AF%BC%20-%20%E5%AE%8C%E6%95%B4%E7%89%88.pdf) |

除常规产测流程外，由于摄像头模组在组装中不可避免会出现偏差，可能导致最终抠图位置出现结构件遮挡；为避免该情况，Turnkey 提供了**自研校准夹具**，搭配 CSK 中集成的校准算法；可根据当前摄像头成像情况，自动调整抠图位置，保证采集的图片清晰，无遮挡；校准后的不良率不超过千分之一；

:::tip

校准夹具与家居参考设计涉及聆思专利，如有需求，可联系商务获取。

:::