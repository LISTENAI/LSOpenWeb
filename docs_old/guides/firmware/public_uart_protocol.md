---
id: public_uart_protocol
title: CSK UART标准串口通信协议文档
slug: /public_uart_protocol
---

| Version | 最后更新时间 | 适用固件版本 |
| :-| :- | :- |
| 1.3 | 2020.03.15 | 4002-3.1.4及以上 |

## 变更记录


| 版本 | 变更内容 | 变更人 | 审核人 | 日期 |
| :- | :- | :- | :- | :- |
| 1.0 | 首版发布 | 朱昊 | 黄凯仪 | 2020-01-08 |
| 1.1 | 更改系统配置描述 | 朱昊 | 黄凯仪 | 2020-02-21 |
| 1.2 | 1. 固定标记修改：命令帧为0xFF 0x00，响应帧（from CSK）为0xF0 0x00。<br />2. 弃用原有0x03版本查询命令帧，新增0x19查询所有版本号命令帧。<br />3. 新增0x05系统休眠命令帧。<br />4. 新增0x16设置ADC增益命令帧。<br />5. 新增0x17设置命令词识别启停命令帧。<br />6. 新增0x18设置UAC通道命令帧。<br />7. 新增0xFF心跳命令帧。<br />8. 弃用原有0x02固件版本反馈响应帧，新增0x04版本查询反馈响应帧。<br />9. 优化0x10识别结果反馈命令数据参数。 | 刘钟蔚 | 朱元恒 | 2020-12-20 |
| 1.3 | 1. 修复0x10 命令帧的数据帧校验<br />2. 修改0x11命令帧的描述<br />3. 修改0x17 命令帧的参数描述 | 朱元恒 | 郝飞虎 | 2021-03-15 |

## 1 概述

CSK作为协处理器时，通过串口通信，为外围模块提供封装的AI能力。此处串口协议我们通过两层协议，来提供完整可靠的CSK API。这两层协议分别为：传输协议、业务协议。传输协议主要负责UART底层的数据进行可靠性保障，而业务协议主要是对CSK能力进行定义，提供可扩展的API。

## 2 传输协议

传输协议，是以接收到的UART原始数据为基础，定义CSK数据帧格式，对流式的二进制数据，进行分帧工作，解析出对应的帧信息，向上传递到业务协议层。传输协议的数据帧格式如下：

<table>
   <tr>
      <th colspan="6" align="center">帧头</th>
      <th colspan="5" align="center">帧数据</th>
   </tr>
   <tr>
      <td colspan="2" align="center">固定标记</td>
      <td colspan="2" align="center">帧长度</td>
      <td align="center">帧编号</td>
      <td align="center">帧头校验</td>
      <td align="center">类型</td>
      <td align="center">地址</td>
      <td align="center">命令字</td>
      <td align="center">命令数据</td>
      <td align="center">帧数据校验</td>
   </tr>
   <tr>
      <td align="center">FTAGL</td>
      <td align="center">FTAGH</td>
      <td align="center">FLENL</td>
      <td align="center">FLENH</td>
      <td align="center">FUID</td>
      <td align="center">LCHK</td>
      <td align="center">TYPE</td>
      <td align="center">REG</td>
      <td align="center">CMD</td>
      <td align="center">DATA</td>
      <td align="center">DCHK</td>
   </tr>
</table>

### 2.1 帧头说明


| 名称 | 长度（byte） | 说明 |
| :- | :- | :- |
| 固定标记 | 2 | 固定为FTAGL = 0x58，FTAGH = 0x46。 |
| 帧长度 | 2 | 由两个字节组成，低字节在前，高字节在后；表示整个命令帧的长度。 |
| 帧序号 | 1 | 表示命令帧的序号，确保短时唯一。 |
| 帧头校验 | 1 | 采用按字节累加和校验；<br />即： （FTAGL + FTAGH + FLENL + FLENH + FUID + LCHK）% 256 = 0 |

### 2.2 帧数据说明


| 名称 | 长度（byte） | 说明 |
| :- | :- | :- |
| 类型 | 1 | 命令帧为0xF0，响应帧（from CSK）为0xFF。 |
| 地址 | 1 | 预留，固定为0x00。 |
| 命令字 | 1 | 表示命令帧/响应帧的命令字。 |
| 命令数据 | X | 不同命令类型，有不同的命令数据结构，具体见：业务协议。 |
| 帧数据校验 | 1 | 采用按字节累加和校验；即：（TYPE + REG + CMD + DATA + DCHK）% 256 = 0 |

## 3 业务协议

数据帧，对于CSK而言，分为命令帧、响应帧，通过帧数据的固定标记，进行区分。

### 3.1 CSK命令帧

外围芯片，可通过“命令帧”来操作CSK，帧数据标记为：0xF0 0x00

### 0x12播放音频0x01系统设置

<table>
   <tr>
      <td align="center">名称</td>
      <td td align="center">发送数据</td>
      <td td align="center">字节数量</td>
      <td td align="center">说明</td>
   </tr>
   <tr>
      <td td align="center">命令类型</td>
      <td td align="center">0x01</td>
      <td td align="center">1字节</td>
      <td>系统设置。 <br/>包括：系统音量设置、就绪信号电平设置、系统闲置休眠时间、识别命令词上传格式等设置。</td>
   </tr>
   <tr>
      <td rowspan="4" td align="center">命令数据</td>
      <td td align="center">VOL</td>
      <td td align="center">1字节</td>
      <td>设置系统音量，仅需开机初始化一次：<br/>【0x00~0x0A】有效值；<br/>【0xFE】默认值：0x08；<br/>【0xFF】历史值；<br/>【其它】无效值，系统将反馈参数错误信息</td>
   </tr>
   <tr>
      <td td align="center">READY</td>
      <td td align="center">1字节</td>
      <td>设置 RDY 电平触发方式，仅需开机初始化一次：<br/>【0x00】低电平；<br/>【0xAA】高电平；<br/>【0xFE】默认值：低电平；<br/>【0xFF】历史值；<br/>【其它】无效值，系统将反馈参数错误信息</td>
   </tr>
   <tr>
      <td td align="center">SLEEP_TIME</td>
      <td td align="center">1字节</td>
      <td>遵循默认值即可。<br/>【0x00】默认值：0x00；<br/>【0x01~0xF0】有效值；<br/>【0xFE】默认值：0x06；<br/>【0xFF】历史值；<br/>【其它】无效值，系统将反馈参数错误信息</td>
   </tr>
   <tr>
      <td td align="center">OUT_FORMAT</td>
      <td td align="center">1字节</td>
      <td>设置识别结果上传内容。启动识别模式，根据输入音频有识别结果后，会根据此参数的不同，上报给上位机不同的格式的识别结果：<br/>【0x00】上报命令词KEYID及唤醒得分和命令词文本，默认值；<br/>【0x01】仅上报命令词KEYID；<br/>【0x02】仅上报命令词文本；<br/>【其它】无效值，系统将反馈参数错误信息</td>
   </tr>
</table>

:::note使用示例

命令帧:设置播音音量为 7 级，RDY 高电平：

```
58 46 0E 00 00 54 F0 00 01 07 AA 0A 00 54
```

响应帧：命令帧正确反馈

```
58 46 0B 00 08 4F FF 00 01 10 F0
```

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)

:::

### 0x02串口设置

设置串口波特率。串口波特率设置完成后，当前帧的状态反馈，如收到正确命令帧等仍是使用之前的波特率发送的（用户可以在接收到正确命令帧之后，更换所需波特率）。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x02</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td>命令数据</td>
        <td>RATE</td>
        <td>1字节</td>
        <td>配置波特率（单位：bps）：<br/> 【0x00】9600；<br/>  【0x01】19200； <br/> 【0x02】57600；<br/>  【0x03】115200默认值； <br/> 【0x04】345600；<br/>  【其它】无效值，系统将反馈参数错误信息；<br/> 需要注意的是，不同版本的默认通信波特率值不一样，详见版本查询命令中相关说明，但打印功能串口的波特率固定为115200。</td>
    </tr>
</table>

:::note使用示例
命令帧:设置波特率为 115200

```
58 46 0B 00 00 57 F0 00 02 03 0B
```

响应帧：命令帧正确反馈

```
58 46 0B 00 02 55 FF 00 01 10 F0
```

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)

:::

### 0x03版本查询（已弃用）

已弃用。

### 0x04状态查询

查询当前系统工作运行状态。模块在收到该命令后，将反馈系统状态数据包，详见响应帧部分说明。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x04</td>
        <td>1字节</td>
        <td></td>
    </tr>
</table>

:::note使用示例

命令帧: 状态查询
58 46 0A 00 00 58 F0 00 04 0C

响应帧：


命令帧正确反馈：

```
58 46 0B 00 04 53 FF 00 01 10 F0
```

运行状态反馈：

```
58 46 0B 00 05 52 FF 00 03 10 EE
```

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)
>
> 查看[响应帧：0x03运行状态反馈](#0x03运行状态反馈)

:::

### 0x05系统休眠（已弃用）

已弃用。

### 0x10停止所有模块

停止当前正在工作的功能模块。包括：语音播放、语音识别、语音唤醒等，停止后进入空闲状态。需要注意的是，通过该指令强制系统结束当前运行的功能模块后，是不会收到当前模块运行结束状态反馈的，例如当前正在播音，发送该命令后，模块不会再发送播音停止状态反馈（正常播音结束后会有该状态反馈）。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x10</td>
        <td>1字节</td>
        <td></td>
    </tr>
</table>

:::note使用示例
命令帧: 停止所有模块

```
58 46 0A 00 00 58 F0 00 10 00
```

响应帧：命令帧正确反馈

```
58 46 0B 00 08 4F FF 00 01 10 F0
```

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)

:::

### 0x11启动识别模式

启动语音识别命令。执行该命令后，系统将根据设置的参数进行语音识别，包括：前置播音、识别模式、识别超时时间等。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x11</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="5" >命令数据</td>
        <td>VID</td>
        <td>2字节</td>
        <td>由两个字节组成，低字节在前，高字节在后；可以且仅支持在启动识别前播放一条提示音，录音播放完成后，将自动启动识别，录音 ID 详见播音命令中的说明。</td>
    </tr>
    <tr>
        <td>MODE</td>
        <td>4 bit</td>
        <td>用来指定识别模式，和BEAM、LOOP共同组成一个字节，MODE占第0-3位。 <br/>【0x00】wakeup模式。该模式仅支持唤醒，唤醒成功后，返回唤醒结果；注意：唤醒是永久等待的，直到唤醒成功； <br/> 【0x01】ASR模式。该模式仅支持识别，识别成功后，返回识别结果；该模式必须指定BEAM。此模式的RT参数可选； <br/>【0x02】oneshot：即一次唤醒，一次识别；唤醒和识别成功都会返回结果；此模式的RT参数可选； <br/>【0x03】多轮交互模式：即一次唤醒后，可以进行多次识别；唤醒和识别成功都会返回结果；此模式会一直进行直到识别超时或者接收到其它需要切换状态的指令才会退出识别； <br/>【其它】无效值，系统将反馈参数错误信息；</td>
    </tr>
    <tr>
        <td>BEAM</td>
        <td>3 bit</td>
        <td>指定识别模式的拾音波束，仅在MODE = 0x01时有效。和MODE、LOOP共同组成一个字节，BEAM占第4-6位。 <br/>【0x00】主要拾音mic1方向音频（0°- 60°）； <br/>【0x01】主要拾音中间方向音频（60°- 120°）； <br/>【0x02】主要拾音mic2方向音频（120°- 180°）；<br/> 【其它】无效值，系统将反馈参数错误信息</td>
    </tr>
    <tr>
        <td>LOOP</td>
        <td>1 bit</td>
        <td>指定识别模式是否循环，和MODE、BEAM共同组成一个字节，LOOP占第7位。 <br/>【0x00】单次。该模式下，识别出结果后，进入空闲状态；<br/> 【0x01】循环。该模式下，识别出结果后，再次启动识别，循环直到接收到其它需要切换状态的指令才会退出识别；</td>
    </tr>
    <tr>
        <td>RT</td>
        <td>1字节</td>
        <td>1 字节，仅用来指定命令词的识别超时，超时后会返回识别超时信息，单位：秒。注意：模块在等待唤醒时会永久等待，因此RT只在识别时起作用。 <br/> 【0x00】默认值，即永久等待，不会超时； <br/> 【0x01~0xFF】有效值；</td>
    </tr>
</table>

:::note使用示例

命令帧:

<table border='0' cellpadding='0' cellspacing='0' width='1404' style={{borderCollapse: 'collapse',tableLayout:'fixed',width:'553pt'}}>
 <col width='172' style={{msoWidthSource:'userset', width:'129pt'}}></col>
 <col width='61' span='2' style={{msoWidthSource:'userset', width:'45.75pt'}}></col>
 <col width='72' span='5' style={{width:'54pt'}}></col>
 <col width='189' style={{msoWidthSource: 'userset', width:'141.75pt'}}></col>
 <col width='129' style={{msoWidthSource: 'userset', width: '96.75pt'}}></col>
 <col width='72' span='6' style={{width:'54pt'}}></col>
 
 <tr height='19' style={{msoHeightSource:'userset', height:'14.4pt'}} id='r1'>
    <td colspan='16' height='19' class='x69' style={{height: '14.4pt'}}>命令帧结构</td>
 </tr>
 <tr height='20' style={{msoHeightSource: 'userset', height: '15pt'}} id='r2'>
<td height='20' class='x69' style={{height: '15pt'}}>字段</td>
<td class='x69'>类型</td>
<td class='x69'>地址</td>
<td class='x69'>命令字</td>
<td colspan='2' class='x69'>提示音ID</td>
<td class='x70' style={{overflow: 'hidden'}}><div style={{marginLeft: '-14px'}}>指定识别模式是否循环</div></td>
<td colspan='3' class='x70'>拾音波束</td>
<td colspan='4' class='x69'>识别模式</td>
<td class='x69'>超时时间</td>
<td class='x69'><span style={{display:'none'}}></span>累加和校验</td>
 </tr>
 

<tr height='19' style={{msoHeightSource: 'userset', height:'14.4pt'}} id='r3'>
<td height='19' class='x69' style={{height:'14.4pt'}}>ID</td>
<td class='x69'>TYPE</td>
<td class='x69'>ADDR</td>
<td class='x69'>CMD</td>
<td colspan='2' class='x69'>VID</td>
<td class='x69'>LOOP</td>
<td colspan='3' class='x69'>BEAM</td>
<td colspan='4' class='x69'>MODE</td>
<td class='x69'>RT</td>
<td class='x69'>DCHK</td>
 </tr>
 <tr height='19' style={{msoHeightSource:'userset', height:'14.4pt'}} id='r4'>
<td height='19' class='x69' style={{height:'14.4pt;'}}>数据位长度</td>
<td class='x69'>1 BYTE</td>
<td class='x69'>1 BYTE</td>
<td class='x69'>1 BYTE</td>
<td class='x69'>1 BYTE</td>
<td class='x69'>1 BYTE</td>
<td class='x71'>Bit7</td>
<td class='x71'>Bit6</td>
<td class='x71'>Bit5</td>
<td class='x71'>Bit4</td>
<td class='x71'>Bit3</td>
<td class='x71'>Bit2</td>
<td class='x71'>Bit1</td>
<td class='x69'>Bit0</td>
<td class='x69'>1 BYTE</td>
<td class='x69'>1 BYTE</td>
 </tr>



 <tr height='19' style={{msoHeightSource: 'userset', height: '14.4pt'}} id='r5'>
<td height='19' class='x69' style={{height: '14.4pt'}}>二进制</td>
<td class='x69'></td>
<td class='x69'></td>
<td class='x69'></td>
<td class='x69'></td>
<td class='x69'></td>
<td class='x69'>1</td>
<td class='x69'>0</td>
<td class='x69'>0</td>
<td class='x69'>1</td>
<td class='x69'>0</td>
<td class='x69'>0</td>
<td class='x69'>1</td>
<td class='x69'>1</td>
<td class='x69'></td>
<td class='x69'></td>
 </tr>
 <tr height='19' style={{msoHeightSource: 'userset', height: '14.4pt'}} id='r6'>
<td height='19' class='x69' style={{height:'14.4pt'}}>示例帧数据(十六进制)</td>
<td class='x69'>0xF0</td>
<td class='x69'>0x00</td>
<td class='x69'>0x11</td>
<td class='x69'>0x00</td>
<td class='x69'>0x01</td>
<td colspan='8' class='x69'>0x93</td>
<td class='x69'>0x14</td>
<td class='x69'>0x57</td>
 </tr>
 <tr height='0' style={{display:'none'}}>
  <td width='172' style={{width:'129pt'}}></td>
  <td width='61' style={{width:'45.75pt'}}></td>
  <td width='61' style={{width:'45.75pt'}}></td>
  <td width='72' style={{width:'54pt'}}></td>
  <td width='72' style={{width:'54pt'}}></td>
  <td width='72' style={{width:'54pt'}}></td>
  <td width='72' style={{width:'54pt'}}></td>
  <td width='72' style={{width:'54pt'}}></td>
  <td width='189' style={{width:'141.75pt'}}></td>
  <td width='129' style={{width:'96.75pt'}}></td>
  <td width='72' style={{width:'54pt'}}></td>
  <td width='72' style={{width:'54pt'}}></td>
  <td width='72' style={{width:'54pt'}}></td>
  <td width='72' style={{width:'54pt'}}></td>
  <td width='72' style={{width:'54pt'}}></td>
  <td width='72' style={{width:'54pt'}}></td>
 </tr>
</table>

```
58 46 0E 00 00 54 F0 00 11 01 00 93 14 57
```

响应帧: 命令帧正确反馈

```
58 46 0B 00 08 4F FF 00 01 10 F0
```

超时后会返回识别超时的响应帧
:::

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)

### 0x12播放音频

语音播放命令，可以播放一个语音序列，最多连续4个。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x12</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4" >命令数据</td>
        <td>VID_1</td>
        <td>2字节</td>
        <td rowspan="4" >有效值：从 1 到 MAX（其中 MAX 取决于实际存储的音频个数，最大取值0xFE）；
         <br/> 
            0x00 作为忽略值及默认值。
            <br/> 
            特殊说明：从第1条开始，如果第N（1&lt;=N&lt;=4）条语音 ID 为 0x00，则该语音之后的 4-N 条语音都将被忽略。
            <br/> 
            由两个字节组成，低字节在前，高字节在后；
        </td>
    </tr>
    <tr>
        <td>VID_2</td>
        <td>2字节</td>
    </tr>
    <tr>
        <td>VID_3</td>
        <td>2字节</td>
    </tr>
    <tr>
        <td>VID_4</td>
        <td>2字节</td>
    </tr>
</table>

:::note使用示例

命令帧：播放两条提示音，第 1 条为 0x01，第 2 条为 0x02，其余的两条为无效值。

```
58 46 12 00 00 50 F0 00 12 01 00 02 00 00 00 00 00 FB
```

响应帧：命令帧正确反馈

```
58 46 0B 00 08 4F FF 00 01 10 F0
```

响应帧：播音结束反馈

```
58 46 0A 00 0D 4B FF 00 20 E1
```

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)
>
> 查看[响应帧：0x20播音结束反馈](#0x20播音结束反馈)

:::

### 0x15设置I2S输出通道

I2S输出通道选择命令，模块支持输出四个通道信号的输出，即可以同时输出四路音频信号，每个通道为16bit数据，并且支持和7路物理音频声道（静音声道，左、右麦克风声道，左、右参考信号声道，左、右降噪后音频声道）的随意映射，实现自由组合输出，满足不同的上位机需求。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x15</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td>命令数据</td>
        <td>CHANNEL</td>
        <td>4字节</td>
        <td>输出通道配置，共4个字节其中：<br/> 1字节：W表示映射到输出通道1的物理声道编号； <br/>1字节：X表示映射到输出通道2的物理声道编号；<br/> 1字节：Y表示映射到输出通道3的物理声道编号； <br/>1字节：Z表示映射到输出通道4的物理声道编号；<br/> 这里：W、X、Y、Z取值的有效范围均为0-6，对应的物理声道编号参见物理声道列表。</td>
    </tr>
</table>

通道号说明：

<table>
    <tr>
        <td>通道号</td>
        <td>通道描述</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>0</td>
        <td>静音通道</td>
        <td>该声道将输出纯零静音数据，一般情况下座位占位或屏蔽通道使用。</td>
    </tr>
    <tr>
        <td>1</td>
        <td>MIC 1 信号</td>
        <td>麦克风1的原始信号。</td>
    </tr>
    <tr>
        <td>2</td>
        <td>MIC 2 信号</td>
        <td>麦克风2的原始信号（如有）。</td>
    </tr>
    <tr>
        <td>3</td>
        <td>参考信号 1</td>
        <td>参考信号1的原始数据。</td>
    </tr>
    <tr>
        <td>4</td>
        <td>参考信号 2</td>
        <td>参考信号2的原始数据（如有）。</td>
    </tr>
    <tr>
        <td>6</td>
        <td>降噪音频 1</td>
        <td>前端算法后的第一路音频，主要用于识别处理。</td>
    </tr>
    <tr>
        <td>6</td>
        <td>降噪音频 2</td>
        <td>前端算法后的第二路音频，主要用于识别处理。</td>
    </tr>
    <tr>
        <td>7</td>
        <td>测试通道</td>
        <td>测试通道数据，主要用于验证是否存在数据丢失等</td>
    </tr>
    <tr>
        <td>8</td>
        <td>云端识别信号</td>
        <td>前端算法后的第三路音频，主要用于云端识别。</td>
    </tr>
</table>

:::note使用示例

命令帧：播放两条提示音，设置音频物理声道 1、2、3、4 设置为对应通道配置
58 46 0E 00 00 54 F0 00 15 01 02 03 04 F1

响应帧：正确命令帧反馈

```
58 46 0B 00 08 4F FF 00 01 10 F0
```

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)

:::

### 0x16设置ADC增益

动态修改ADC的增益，分别对就MIC1、MIC2、REF1、REF2的增益。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x16</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4" >命令数据</td>
        <td>MIC_1</td>
        <td>1字节</td>
        <td rowspan="2">MIC信号增益，分为10级.</td>
    </tr>
    <tr>
        <td>MIC_2</td>
        <td>1字节</td>
    </tr>
    <tr>
        <td>REF_1</td>
        <td>1字节</td>
        <td rowspan="2">回采信号增益，分为10级.</td>
    </tr>
    <tr>
        <td>REF_2</td>
        <td>1字节</td>
    </tr>
</table>

:::note使用示例
命令帧：设置 ADC 增益为 10 级(2 路 mic 增益 8，2 路回采增益为1)

```
58 46 0E 00 00 54 F0 00 16 08 08 01 01 E8
```

响应帧：命令帧正确反馈

```
58 46 0B 00 08 4F FF 00 01 10 F0
```

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)

:::

### 0x17设置命令词识别的启停

控制系统是否进入识别模式，即跳过唤醒，直接进入命令词的识别。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x17</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4" >命令数据</td>
        <td>MODE_CTRL</td>
        <td>1字节</td>
        <td>控制命令词识别的进入、退出。<br/> 注：只有在命令词识别状态下，才会有CAE第三路音频输出； <br/>【0x00】进入命令词识别状态； <br/>【0x01】退出命令词识别状态； 其他值无效；</td>
    </tr>
    <tr>
        <td>DEF</td>
        <td>1字节</td>
        <td rowspan="3" >保留字段，固定为 0x00</td>
    </tr>
    <tr>
        <td>DEF</td>
        <td>1字节</td>
    </tr>
    <tr>
        <td>DEF</td>
        <td>1字节</td>
    </tr>
</table>

:::note使用示例
命令帧：进入命令词识别状态

```
58 46 0E 00 00 54 F0 00 17 00 00 00 00 F9
```

响应帧：命令帧正确反馈

```
58 46 0B 00 08 4F FF 00 01 10 F0
```

:::

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)

### 0x18设置UAC通道

当固件设置为通过UAC输出音频时，该命令可用来设置UAC输出的音频通道号。UAC仅输出一个通道音频。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x18</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td>命令数据</td>
        <td>CHANNEL</td>
        <td>1字节</td>
        <td>【0x00】静音输出，即输出数据为0； <br/>【0x01】降噪后第一路音频； <br/>【0x02】降噪后第二路音频；<br/> 【0x03】降噪后第三路音频（用于云端识别）；<br/> 【0x04】测试用的音频（用于测试UAC传输的完整性）；</td>
    </tr>
</table>

:::note使用示例
命令帧：输出第三路音频，用于云端识别

```
58 46 0B 00 00 57 F0 00 18 03 F5
```

响应帧：命令帧正确反馈

```
58 46 0B 00 08 4F FF 00 01 10 F0
```

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)

:::

### 0x19查询所有版本号

查询固件、算法、资源等版本号，需要查询哪些版本号，可通过参数配置。各版本号分开响应帧分开输出。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x19</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td>命令数据</td>
        <td>MODUL</td>
        <td>1字节</td>
        <td>需要查询版本号的模块； <br/>【0x00】查询所有模块的版本号，各模块的版本号分开响应； <br/>【0x01】查询固件版本号； <br/>【0x02】查询CAE的版本号； <br/>【0x03】查询miniESR的版本号；</td>
    </tr>
</table>

:::note使用示例
命令帧：查询固件 CAE 版本号

```
58 46 0B 00 00 57 F0 00 19 02 F5
```

响应帧：命令帧正确反馈

```
58 46 0B 00 08 4F FF 00 01 10 F0
```

响应帧：版本查询的反馈

```
58 46 4B 00 24 F3 FF 00 04 xxxx
```

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)
>
> 查看[响应帧：0x04版本查询的反馈](#0x04版本查询的反馈)

:::

### 0xFF心跳

CSK接收的心跳指令后，返回一个心跳“响应帧”，表示工作正常。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0xFF</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td>命令数据</td>
        <td>DEF</td>
        <td>1字节</td>
        <td>保留字段，固定为 0x00</td>
    </tr>
</table>

:::note使用示例
命令帧：心跳包

```
58 46 0B 00 00 57 F0 00 FF 00 11
```

响应帧：命令帧正确反馈

```
58 46 0B 00 08 4F FF 00 01 10 F0
```

> 查看[响应帧：0x01 命令帧反馈](#0x01命令帧反馈)

:::

---

### 3.2 CSK响应帧

CSK通过响应帧，向外围模块发送CSK的相关信息；帧数据标记为：0xFF 0x00。

### 0x01命令帧反馈

系统正常启动、接收到命令帧后，CSK会发送本响应；

<table>
   <tr>
      <td>名称</td>
      <td>发送数据</td>
      <td>字节数量</td>
      <td>说明</td>
   </tr>
   <tr>
      <td>命令类型</td>
      <td>0x01</td>
      <td>1字节</td>
      <td></td>
   </tr>
   <tr>
      <td>命令数据</td>
      <td>STATE</td>
      <td>1字节</td>
      <td>【0x00】系统初始化成功（系统正常启动后，会主动发送该响应）<br/>【0x10】收到正确的命令<br/>【0x11】数据帧格式错误<br/>【0x12】数据帧参数错误</td>
   </tr>
</table>

> 查看[命令帧：0x01 系统设置](#0x01系统设置)

### 0x02固件版本（已弃用）

已弃用。

### 0x03运行状态反馈

系统运行状态命令帧的反馈。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x03</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td>命令数据</td>
        <td>STATE</td>
        <td>1字节</td>
        <td>【0x00】空闲和就绪状态 <br/>【0x10】语音识别状态 <br/>【0x11】语音播放状态 <br/>【0x12】语音合成状态 <br/>【0x13】系统休眠（低功耗）状态</td>
    </tr>
</table>

> 查看[命令帧：0x01 系统设置](#0x01系统设置)

### 0x04版本查询的反馈

针对版本查询命令帧（0x19）的响应，如果查询的是所有模块的版本，那么会分多次反馈各个模块的版本。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x04</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td>命令数据</td>
        <td>TYPE</td>
        <td>1字节</td>
        <td>【0x01】固件版本号；<br/> 【0x02】CAE的版本号；<br/> 【0x03】miniESR的版本号；</td>
    </tr>
    <tr>
        <td></td>
        <td>VER_1</td>
        <td>32字节</td>
        <td>基础固件/算法版本号</td>
    </tr>
    <tr>
        <td></td>
        <td>VER_2</td>
        <td>32字节</td>
        <td>固件构建号/资源版本号（暂时全部为0x00）</td>
    </tr>
</table>

> 查看[响应帧：0x19查询所有版本号](#0x19查询所有版本号)

### 0x10识别结果反馈

语音识别系统处理结束，将识别结果回传上位机，根据系统设置命令的【OUT_FORMAT】 段设置，输出不同格式的结果。

OUT_FORMAT = 0时：

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x10</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="3" >命令数据</td>
        <td>ID</td>
        <td>2字节</td>
        <td>识别结果编号，取值范围：0~65535，该值代表对应的识别结果的编号，可通过该值获知是哪个命令被触发了。</td>
    </tr>
    <tr>
        <td>SCORE</td>
        <td>2字节</td>
        <td>命令词分数</td>
    </tr>
    <tr>
        <td>TEXT</td>
        <td>N字节，不超过200</td>
        <td>识别结果名称，最大不超过200个字节（100个字），该值表示识别结果对应的文本。注意：此文本编码固定为UTF-8格式。</td>
    </tr>
</table>

OUT_FORMAT = 1时：

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x10</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td>命令数据</td>
        <td>ID</td>
        <td>2字节</td>
        <td>识别结果编号，取值范围：0~65535，该值代表对应的识别结果的编号，可通过该值获知是哪个命令被触发了。</td>
    </tr>
</table>

OUT_FORMAT = 2 时：

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x10</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td>命令数据</td>
        <td>TEXT</td>
        <td>N字节，不超过200</td>
        <td>识别结果名称，最大不超过200个字节（100个字），该值表示识别结果对应的文本。注意：此文本编码固定为UTF-8格式。</td>
    </tr>
</table>

### 0x11识别异常反馈

识别异常状态反馈，包括：超时、拒识、错误等信息。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x11</td>
        <td>1字节</td>
        <td></td>
    </tr>
    <tr>
        <td>命令数据</td>
        <td>ERROR</td>
        <td>1字节</td>
        <td>【0x01】识别超时：用户在一段时间内未说话，出现反应超时；用户说话的时间超过设定的时间，出现识别超时；<br/> 【0x10】引擎错误：引擎内部出现错误，通常用户是接收不到此状态的，如果确实接收到了，请复位模块；</td>
    </tr>
</table>

### 0x20播音结束反馈

播音结束后（注意，仅对播音指令有效，并且只有正常播放结束，才会有该状态信息反馈。识别或唤醒等前置提示音播放，或者播音强制结束的都不会有该状态反馈）。

<table>
    <tr>
        <td>名称</td>
        <td>发送数据</td>
        <td>字节数量</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>命令类型</td>
        <td>0x20</td>
        <td>1字节</td>
        <td></td>
    </tr>
</table>

> 查看[命令帧：0x12播放音频](#0x12播放音频)
