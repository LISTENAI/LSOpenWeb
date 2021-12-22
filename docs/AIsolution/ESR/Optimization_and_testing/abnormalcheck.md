---
sidebar_label: 效果测试异常排查
sidebar_position: 4
---

# 效果测试异常排查

## 1. 术语和定义

### 1.1. 测试术语及定义

* **混响**：一种声学特性，声波在室内传播时，要被墙壁、天花板、地板等障碍物反射，每反射一次都要被障碍物吸收一些。这样，当声源停止发声后，声波在室内要经过多次反射和吸收，最后才消失。这种现象叫做混响，从声音发出经多次反射后直至听不到声音为止的这段时间叫做混响时间。
* **分贝**：声压级的单位，大约等于人耳通常可觉察响度差别的最小分度值，单位为dB，。通常使用声压计（也叫声级计、分贝仪）作为检测仪器，声压计通常有A、C、Z三种计权，dBA表示的就是A计权下的声压级，本文中所有分贝均指dBA。
* **唤醒词**：相当于给设备起的名字，说话人通过说出唤醒词，唤醒语音模块，使其进入指令识别状态。例如：模块的唤醒词为“小飞小飞”，则使用者只要说出“小飞小飞”即可唤醒模块。
* **命令词**：智能语音产品制定的控制指令，多数是对产品功能控制的文本描述，用户说出命令词，系统识别成功后，配合设备的控制系统，实现功能的触发，如“打开空调”。

* **信噪比（SNR）**：本文统一指人声分贝值减去环境噪声分贝值的差值 

* **场景**：指目标应用产品所处的噪声、混响环境，可以从空间大小、混响环境、噪音类型进行区分，空间分布主要有客卧、厨房、卫生间、阳台；根据时间段不同，对于设备而言的环境噪音环境也会不同，例如家居环境主要分为白天（安静），白天（常噪），傍晚聚集（高噪），晚间睡眠场景等。

  *说明：以上人声噪音中，均不包含唤醒词和命令词相关音频，统计过程中，不将其作为误唤醒/误识别进行处理。*

* **测试集**：测试音频集，根据测试用途分为唤醒测试集、识别测试集、误唤醒测试集。





### 1.2. 效果指标相关定义

* **唤醒率**：不同场景下，设备待唤醒状态下获取唤醒词的成功次数/总播放次数*100%。如没有回声消除功能，测试过程中，需要等播报结束再进行唤醒。

* **误唤醒频度**：模拟用户实际使用场景，将待测设备调至待命状态下，在不含唤醒词的前提下，记录规定时间段内设备被误唤醒的次数。

* **唤醒响应时间**：用户说出唤醒词，到设备成功唤醒并给出唤醒应答语的时间间隔，统计区间为：唤醒词最后一个字说完至设备唤醒应答语第一个字播出前

* **识别率**：设备被唤醒，进入识别状态后，获取命令词识别成功的次数/总播放唤醒次数*100%，如没有回声消除功能，测试过程中，需要等播报结束再说命令词。

* **串扰**：设备接收到相关语音内容，有识别结果但识别成其他集内词。串扰主要分为集内串扰和集外串扰。关于串扰及相关指标的关系，如下图示意，A（A1∪A2∪A3）为集内词测试集，B（B1∪B2）为集外词测试集，A∪B构成测试全集，本文档主要讨论集内串扰率。


​																								A1——正确识别

​																								A2——有识别结果但识别成其他集内词A3——无识别结果  

​																								B1——无识别结果

​																								B2——有识别结果

​																								集内正确率计算方式：A1 / A × 100% 

​																								集内串扰率计算方式：A2 / A × 100% 







## 2. 通用问题定位教程

硬件 -> 软件 -> 测试条件。

### 2.1. 硬件

效果测试开始前，设备声学结构必须确认ok

| 硬件检查项                                                   | 要求                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1、麦克风阵列频率响应一致性               | 麦克风收音音量大小要一致;<br/>麦克风频响要一致;<br/>不要有异常的电流音·不要有丢音;<br/>7.5K频率以上不要有明显的衰减。  |
| 2、麦克风增益                             | 模拟麦要调节模拟增益，调节数字增益意义不大;<br/>增益调到，收音尽可能的大，但不要截幅，为准。  |
| 3、整机气密性                             | 堵上麦克风与不堵上麦克风收音的音量差值，要有15db以上  |
| 4、整机喇叭                               | 整机设备喇叭需通过硬件测试，保证喇叭在各频段表现无明显谐振，输出功率需要满足最大播放音量，建议不超过100dBA。  |
| 5、录制10分钟唤醒识别音频、10分钟打断唤醒音频确认录音是否正常  |                                               /               |
| 6、连续录制1小时以上的音频，观察是否会出现中断或异常  |                  /                                            |
| 7、拔掉麦克风，接2.2K电阻，录1min音频     | 底噪要求小于-60db                         |

:::note
麦克风阵列频率响应一致性验证操作方法：在安静实验室（背景噪声35dBA），音箱在0.8m处播放白噪声（保障白噪声到MIC处的音量>=70db），使用被测设备进行录音，使用Adobe Audition分析音频。打开Adobe Audition上的振幅统计查看各个声道的平均RMS振幅，若相差在±3dB 范围内， 可认为麦克风阵列频率响应是一致。   
:::

:::note
麦克风增益验证操作方法：麦克风增益需要满足在设备最大音量播放歌曲等音频时，麦克风原始录音各个通道都不截幅（各MIC通道、回采信号通道），在此条件上尽可能的加大增益。   
:::

:::note
整机气密性验证操作方法：在安静实验室（背景噪声35dBA），使用音箱播放白噪声（保障白噪声到MIC处的音量>=70db），统计用橡皮泥堵住麦克风进声孔之后，麦克风拾音的下降量。在同一环境下进行录音，使用Adobe Audition统计正常录音（未使用橡皮泥堵住麦克风进声孔）与用橡皮泥堵住麦克风录音的振幅，计算出拾音下降量，拾音下降量在15dB以上说明麦克风的气密性是良好的（拾音下降量大，说明密封良好；拾音下降量小，说明有漏音或整机中出现谐振等现象，需要进行硬件结构排查）。   
:::



**由硬件带来的效果异常：**

| **测试结果异常辨别：**                                       | **排除思路：**                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1.近距离唤醒识别比远距离唤醒识别效果低    | 设备麦克风开孔是否较低，是否有遮挡，麦克风开孔较低对于近距离唤醒识别有影响  |
| 2. 安静场景唤醒率/识别率每次测试结果不一致，相差超过5%  | 因硬件导致的效果异常，如结构未达到声学评估标准，若是，则需要对设备声学结构进行调整；<br/>唤醒测试集中每句唤醒词间隔太短，导致设备没有反应过来，或未成功打断设备应答语，若是，延长唤醒间隔  |





### 2.2. 软件

| 软件检查项                      | 要求                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| 1、音频检验  | 可录制原始音频;<br/>可录制降噪后音频;<br/>支持长时间录制以上两组音频  |
| 2、日志      | 唤醒日志打印正确，包含唤醒时间、唤醒次数、唤醒词;<br/>唤醒日志中角度打印正确;<br/>识别日志打印正确，包含识别结果、识别时间  |
| 3、回声消除  | 设备本机可播音，例如push音频文件播放;<br/>设备可调节音量;<br/>本机播音的同时，进行唤醒日志的打印及降噪后音频的录制  |

辅助工具：[LStudio下载](https://castor.iflyos.cn/castor/v3/lstudio/download)


**由软件带来的效果异常：**

| **测试结果异常辨别**               | **排除思路**                                               |
| ------------------------------------ | ------------------------------------------------------------ |
| 1.唤醒率大于100%  | 唤醒日志打印不准确，重复打或者少打，或者872和csk打的唤醒次数对不上；若是需要排查固件异常;<br/>设备出现了误唤醒，会导致唤醒率大于100%，若是，需更换噪音内容；若是在安静状态也容易发生误唤醒，需要先调整唤醒门限再进行测试  |



### 2.3. 测试条件

| 测试条件检查项                                    | 要求                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ |
| 1、环境混响时间                | 小混响环境的混响时间小于 0.3s（卧室（木质地板））;<br/>中混响环境的混响时间一般在 0.3s~0.7s（客厅、阳台、家电卖场环境、办公室场景、包厢场景、电视场景、聚会场景）;<br/>大混响环境的混响时间大于 0.8s（厨房、浴室、大仓库场景）  |
| 2、房间底噪                    | 高端声学实验室底噪通常在20db以下;<br/>通用型声学实验室的底噪为30db-40db;<br/>家居环境昼间噪音上限为55db-60db;<br/>夜间噪音上限为45db-50db;<br/>办公区办公期间噪音上限约为60db-65db;<br/>午休期间噪音上限约为45db~55db  |
| 3、测试噪声                                       | 在进行语音方案效果的验证测试时，需分析产品实际应用场景中环境噪音的主要内容。通常环境噪 音不会只包含一种，因此需要确认各种类型的噪音在整体噪音中的时长占比。  |
| 4、确认词表与测试音频是否对应  | 唤醒、离线命令词、在线命令词、通用技能说法  |
| 5、筛查测试音频                | 采集回来的音频需要做筛选，剔除语速不正常（过快过慢）、发音不清楚的音频  |
| 6、环境搭建                                       | 在准备测试方案之前，可根据被测物的实际使用场景，进行相 关环境搭建。  |
| 7、信噪比                      | 在保证实验室安静的前提下，分别单独测量唤醒/命令词测试集，与噪音语料的分贝值，唤醒/命令词分贝值减去噪音分贝值之差即为信噪比，录音信号比电路底噪，要尽可能的大  |
| 8、误唤醒测试集                | 误唤醒测试集不包含唤醒词和命令词相关音频，并且需要每个季度更新一次，同步当下最新的影视剧/综艺等资源  |
| 9、唤醒/识别测试集             | 建议采集高保真音频，格式为16K16bit单通道。录制时要求语速正常、男女比例各半，音频数量必须在300句以上。  |

具体可参考文档[前端声学综合效果评估及测试标准__v1.1_](https://open.listenai.com/resource/open/doc_resource%2F%E8%AF%AD%E9%9F%B3%E4%BA%A4%E4%BA%92%E6%95%88%E6%9E%9C%E6%B5%8B%E8%AF%95%2F%E5%89%8D%E7%AB%AF%E5%A3%B0%E5%AD%A6%E7%BB%BC%E5%90%88%E6%95%88%E6%9E%9C%E8%AF%84%E4%BC%B0%E5%8F%8A%E6%B5%8B%E8%AF%95%E6%A0%87%E5%87%86_v1.1.pdf)

 

**由测试条件带来的效果异常：**

| **测试结果异常辨别：**                                       | **排除思路：**                                               |
| ------------------------------------------------------------ | :----------------------------------------------------------- |
| 1.安静场景（不加外噪，仅实验室底噪）唤醒率低于80%，识别率低于60%  | 1.1.环境是否发生变化（周围有人讲话、外界有噪音），检查重新测量底噪和混响是否满足要求<br/>1.2.唤醒测试集快语速或慢语速占比过高，若是，需要重新制作测试集<br/>1.3. 唤醒测试集中每句唤醒词间隔太短，导致设备没有反应过来，或未成功打断设备应答语，若是，需要在测试工具调整静音时间  |
| 2.安静场景唤醒率每次测试结果不一致，相差超过5%  | 2.设备本身不稳定，如结构未达到声学评估标准，若是，需要重新检测声学结构  |

:::tip 注意

所有效果测试，都是基于“录音正常”的基础上，如果收音有缺陷，一定要先解决收音问题后再验证效果。

影响唤醒率、识别率的因素很多，例如麦克风的选型、结构的合理性、唤醒词的选择、实际运用场景等

:::







## 3. 不同异常问题的快速定位指南

### 3.1. 唤醒效果差

**异常辨别：**

1、 安静场景（不加外噪，仅实验室底噪）唤醒率低于80%

2、 安静场景唤醒率每次测试结果不一致，相差超过5%

| 可能因素：                                         | 解决办法：                                                   |
| -------------------------------------------------- | ------------------------------------------------------------ |
| 1.声学结构有问题                                   | 1.重新拿给硬件工程师排查                                     |
| 2.资源搞错了                                       | 2.确认版本信息                                               |
| 3.测试集中存在过快语速、过慢语速                   | 3.用纯正常语速在安静场景下测一遍，若效果很好，则语速是关键因素；若效果依然差，进行下一步（必要时需要人耳筛选） |
| 4.测试集中快语速/慢语速占比过大                    | 4.用纯普通话在安静场景下测一遍，若效果很好，则口音是关键因素（必要时需要人耳筛选） |
| 5.测试集中音频发音不准、口音占比太大、口音种类较多 | 5.若存在死都唤不醒的音频，确认音频内容，若发音不准，剔除坏数据；若发音没问题，提交研发排查，此类音频，在摸底测试阶段可暂时剔除出测试集 |
| 6.识别模式唤醒率低、或纯唤醒模式唤醒率低           | 6.在纯识别模式和纯唤醒模式下各测一次唤醒，对比差异           |
| 7.唤醒日志打印不准确，重复打或者少打               | 7.需要修改设备日志打印方式                                   |





### 3.2. 误唤醒很高

**异常辨别：**

1、 安静场景（不加外噪，仅实验室底噪）测试24h，误唤醒大于3次

2、 加噪场景测试24h,误唤醒大于10次

| 可能因素：               | 解决办法：                       |
| ------------------------ | -------------------------------- |
| 1.机器声学结构有问题     | 1.重新拿给硬件工程师排查         |
| 2.版本搞错了、没刷进去   | 2.确认版本信息、重刷固件         |
| 3.误唤醒测试集中有相似词 | 3.听下误唤醒发生时间段的音频内容 |
| 4.阈值太高               | 4.需要重新调整唤醒门限再进行测试 |





### 3.3. 离线识别效果差

**异常辨别：**

1、 安静场景（不加外噪，仅实验室底噪）识别率（句正确率）小于50%

2、 安静场景唤醒率每次测试结果不一致，相差超过5%

3、 出现大量识别为空

| 可能因素：                                         | 解决办法：                                                   |
| -------------------------------------------------- | ------------------------------------------------------------ |
| 1.资源搞错了                                       | 1.确认版本信息                                               |
| 2.测试集中存在过快语速、过慢语速                   | 2.用纯正常语速在安静场景下测一遍，若效果很好，则语速是关键因素；若效果依然差，进行下一步（必要时需要人耳筛选） |
| 3.测试集中快语速/慢语速占比过大                    | 3.用纯普通话在安静场景下测一遍，若效果很好，则口音是关键因素（必要时需要人耳筛选） |
| 4.测试集中音频发音不准、口音占比太大、口音种类较多 | 4.若存在总也识别不了的音频，确认音频内容，若发音不准，剔除坏数据；若发音没问题，提交研发排查，摸底测试阶段可暂时剔除此类音频 |
| 5.设备本身不稳定                                   | 5.如结构未达到声学评估标准，若是，需要重新检测声学结构       |
| 6.出现大量识别为空，一般有三种情况                 | 6.1.唤醒率低，大量语料未唤醒； 6.2. 设备唤醒后未识别；需要提高唤醒词的音量，并检查是否命令词未对设备应答语成功打断。 6.3.另外有一种极端情况是，识别测试集中的命令词，设备本身不支持。 |

:::tip 注意

先确认唤醒有没有问题

:::





### 3.4. 在线识别效果差

| 可能因素：                                         | 解决办法：                                                   |
| -------------------------------------------------- | ------------------------------------------------------------ |
| 1.音频内容错误，跟标注对不上                       | 1.总是识别错误的那些音频，需要听下音频内容是否发音错误       |
| 2.测试环境网络不稳定、设备联网不稳定               | 2.检查测试环境网络，若网络没问题，检查设备日志是否发生了联网中断 |
| 3.设备自身运行不稳定                               | 3.挂机联网1h，不间断进行语音交互，检查设备运行是否异常       |
| 4.测试集中存在过快语速、过慢语速                   | 4.用纯正常语速在安静场景下测一遍，若效果很好，则语速是关键因素；若效果依然差，进行下一步（必要时需要人耳筛选） |
| 5.测试集中快语速/慢语速占比过大                    | 5.用纯普通话在安静场景下测一遍，若效果很好，则口音是关键因素（必要时需要人耳筛选） |
| 6.测试集中音频发音不准、口音占比太大、口音种类较多 | 6.若存在总也识别不了的音频，确认音频内容，若发音不准，剔除坏数据；若发音没问题，提交研发排查，摸底测试阶段可暂时剔除此类音频 |





### 3.5. 安静场景效果很差

| 可能因素：             | 解决办法：                                         |
| ---------------------- | -------------------------------------------------- |
| 1.机器声学结构有问题   | 1.重新拿给硬件工程师排查                           |
| 2.版本搞错了、没刷进去 | 2.确认版本信息、重刷固件                           |
| 3.版本稳定性有问题     | 3.测试稳定性至少2h，看是否会出现连续唤醒后唤不醒   |
| 4.词表搞错了           | 4.跟项目经理/产品经理确认词表的正确性              |
| 5.音频本身说错了       | 5.确认音频内容是不是跟标注有差异                   |
| 6.噪声干扰             | 6.确认机器是否有自噪，环境底噪是否太大（超过45db） |





### 3.6. 加噪场景效果下降很大

| 可能因素：                                             | 解决方法：                                                   |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| 1.噪声难度太高                                         | 1.更换噪声（尽量使用统一的噪声）                             |
| 2.噪声整体音量起伏太大                                 | 2.对噪声做能量规整，并多次测量噪声前中后音量（噪声不要太长，建议30min以内，否则不好控制音量） |
| 3.噪声/人声音量测量的不准，导致信噪比比预期低          | 3.噪声音量定标                                               |
| 4.有其他噪声引入（测试场地底噪，机器自噪，机器震动等） | 4.关闭机器自噪（mute掉设备tts、拔掉设备喇叭、关闭电控等）    |





### 3.7. 串扰率高

**异常辨别：**

1、 安静场景（不加外噪，仅实验室底噪）串扰率大于5%

2、 单个命令词的串扰率大于10%

| 可能因素：      | 解决办法：                                                   |
| --------------- | ------------------------------------------------------------ |
| 1.音频发音相似  | 1.确认音频发音，剔除坏数据，如果是确认发音无误，提交研发排查，摸底测试时可以剔除此类音频 |
| 2.日志打印错误  | 2.识别结果本身是正确的，但打印成了另一句命令词，检查日志打印是否正常 |
| 3.若串扰率＞10% | 3.命令词门限需要重新调整后再进行测试                         |





### 3.8. 客观测试每次结果差异大

| 可能因素：            | 解决方法：                                                   |
| --------------------- | ------------------------------------------------------------ |
| 1.测试场地变换        | 1.固定使用一个测试场地                                       |
| 2.测试环境底噪变化    | 2.注意测试全天测试环境的底噪变化，在稳定的时间段测试         |
| 3.噪声/人声音量不一致 | 3.噪声和人声的音量要固定（无论是偏大还是偏少，用一致的就行） |
| 4.噪声和人声不同步    | 4.尽量采用声卡播放，可以实现多个音频同步播放；无法同步播放的，尽量保持一致 |
| 5.测试集不一致        | 5.测试集保持一致，若必须更换测试集，先在同一个测试集上验证效果一致性 |
| 6.版本不一致          | 6.版本保持一致，若必须更换版本，可使用两台设备分别刷不同的版本，在同一个场景下同时测试对比一致性 |


:::tip 注意

若以上方法都无法解决，加大测试集体量（500句以上），重复测试两遍；若效果依然存在差异，录音提交给研发分析。

:::





### 3.9. 客户现场与聆思的客观测试差异大

| 可能因素：                                                   | 解决方法：                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1.测试环境不一致（包括底噪、混响、房间大小、房间内是否有堆积杂物） | 1.采用相同的测试场景（距离/角度/高度/信噪比）                |
| 2.测试场景不一致                                             | 2.在更符合产品真实使用场景的场地进行主观测试，客观测试尽量复制主观测试环境 |
| 3.使用的噪声不一致                                           | 3.噪声保持一致                                               |
| 4.使用的测试集语速比例、口音比例不一致                       | 4.测试集规范一致                                             |
| 5.使用的设备不一致（结构对效果影响很大）                     | 5.尽量使用相同状态和型号的整机进行测试                       |
| 6.分贝测量不一致（包括分贝仪差异、测试方式差异、人为操作误差等） | 6.采有同样的测量分贝方式，校准分贝仪差距                     |
| 7.版本不一致                                                 | 7.版本信息对齐                                               |

:::tip 注意

客户现场客观测试与聆思客观测试差异允许有5%以内的偏差

:::





### 3.10. 客观测试与主观测试差异大

| 可能因素：                                       | 解决方法：                                                   |
| ------------------------------------------------ | ------------------------------------------------------------ |
| 1.测试场景不一致（混响大小、底噪大小、场地大小） | 1.在更符合产品真实使用场景的场地进行主观测试，客观测试尽量复制主观测试环境 |
| 2.设备摆放位置不一致                             | 2.尽量按照相同的方式摆放测试设备（参考验收场景）             |
| 3.人声/噪声音量不一致                            | 3.在主观测试现场测试人声、噪声分贝，若人声太小或噪声太大，进行相应的调整 |
| 4.口音差异大                                     | 4.若主观测试口音比较偏，引导其修正口音或换人测试；若口音正常，录音提供给研发分析 |
| 5.语速差异大                                     | 5.若主观测试人员过快或过慢，引导其修正                       |
| 6.测试人员发音不标准                             | 6.确认发音是否清晰                                           |
| 7.主观测试环境信噪比变化大                       | 7.在环境稳定的时候进行主观测试，确保底噪稳定                 |
| 8.设备不一致                                     | 8.确认设备是否一致                                           |
| 9.版本不一致                                     | 9.确认版本是否一致                                           |





### 3.11. 不同异常引起程序崩溃

| 可能因素：                          | 解决方法：                                                   |
| ----------------------------------- | ------------------------------------------------------------ |
| 1.内存泄漏引起崩溃                  | 1.测试需要考虑到一些边界值等，并配置相关环境满足验证，尽早发现问题 |
| 2.反复配网出现崩溃                  | 2.测试过程中，及时记录log、录音或内存监控等，提供给研发进行分析 |
| 3.唤醒、识别等功能交互出现崩溃      |         修复固件缺陷                                                     |
| 4.语音交互次数过多引起相关SDK等崩溃 |                 修复固件缺陷                                             |

 



### 3.12. 断电或重启引起程序异常

| 可能因素：                                                   | 解决方法：                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1.开机数分钟内，不同时间点异常断电、断网，引起信息丢失、无法配网或崩溃等 | 1.配置相关环境满足验证条件，尽早发现问题                     |
| 2.正常使用中，不同时间点异常断电、断网，引起信息丢失、无法配网或崩溃等 | 2.测试过程中，及时记录log、录音或内存监控等，提供给研发进行分析 |
| 3.在弱网条件下使用时，出现配网异常或崩溃等                   |           修复固件缺陷                                                   |
| 4.升级各个阶段，异常断电、断网，引起信息丢失、无法配网或崩溃等 |               修复固件缺陷                                               |
