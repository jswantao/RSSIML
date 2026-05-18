**表2. XRF55的动作类别。我们选取了16个视频数据集和19个射频传感数据集的动作。**

| 索引                                               | 类别 (中文)                                   |
| :------------------------------------------------- | :-------------------------------------------- |
| **人类-物体交互 (Human-Object Interaction)** |                                               |
| 1                                                  | 搬运重物 (Carrying Weight)                    |
| 2                                                  | 拖地 (Mopping the Floor)                      |
| 3                                                  | 裁剪 (Cutting)                                |
| 4                                                  | 戴帽子 (Wearing Hat)                          |
| 5                                                  | 使用电话 (Using a Phone)                      |
| 6                                                  | 扔东西 (Throwing Something)                   |
| 7                                                  | 把某物放在桌子上 (Put Something on the Table) |
| 8                                                  | 穿衣服 (Put on Clothing)                      |
| 9                                                  | 捡起 (Picking)                                |
| 10                                                 | 粉刷 (Painting)                               |
| 11                                                 | 抽烟 (Smoking)                                |
| 12                                                 | 舔 (Licking)                                  |
| 13                                                 | 刷牙 (Brushing Teeth)                         |
| 14                                                 | 吹干头发 (Blow Dry Hair)                      |
| 15                                                 | 洗头 (Brush Hair)                             |
| **人际互动 (Human-Human Interaction)**       |                                               |
| 16                                                 | 握手 (Shake Hands)                            |
| 17                                                 | 拥抱 (Hugging)                                |
| 18                                                 | 手触碰某物/某人 (Hand Something to Someone)   |
| 19                                                 | 踢某人 (Kick Someone)                         |
| 20                                                 | 用某物击中某人 (Hit Someone with Something)   |
| 21                                                 | 掐某人的脖子 (Choke Someone's Neck)           |
| 22                                                 | 推某人 (Push Someone)                         |
| **健身 (Fitness)**                           |                                               |
| 23                                                 | 体重深蹲 (Body Weight Squats)                 |
| 24                                                 | 太极 (Tai Chi)                                |
| 25                                                 | 弓步 (Bowing)                                 |
| 26                                                 | 举重 (Weightlifting)                          |
| 27                                                 | 呼啦圈 (Hula Hooping)                         |
| 28                                                 | 跳爆竹/开合跳 (Jump Rope)                     |
| 29                                                 | 跳跃击掌 (Jumping Jack)                       |
| 30                                                 | 高抬腿 (High Leg Lift)                        |
| **身体动作 (Body Motion)**                   |                                               |
| 31                                                 | 挥手 (Waving)                                 |
| 32                                                 | 拍手 (Clap Hands)                             |
| 33                                                 | 跌倒在地 (Fall on the Floor)                  |
| 34                                                 | 跳跃 (Jumping)                                |
| 35                                                 | 跑步 (Running)                                |
| 36                                                 | 坐着 (Sitting Down)                           |
| 37                                                 | 站起来 (Standing Up)                          |
| 38                                                 | 鞠躬 (Bowing)                                 |
| 39                                                 | 行走 (Walking)                                |
| 40                                                 | 伸展 (Stretch Oneself)                        |
| 41                                                 | 把手放在肩膀上 (Put on Shoulder)              |
| 42                                                 | 弹尤克里里 (Playing Ukulele)                  |
| 43                                                 | 打鼓 (Playing Drum)                           |
| **人机交互 (Human-Computer Interaction)**    |                                               |
| 45                                                 | 跺脚 (Stomping)                               |
| 46                                                 | 摇头 (Shaking Head)                           |
| 47                                                 | 阅读 (Reading)                                |
| 48                                                 | 画圆圈 (Draw Circles)                         |
| 49                                                 | 画十字 (Draw a Cross)                         |
| 50                                                 | 推拉 (Pushing)                                |
| 51                                                 | 推拉 (Pulling)                                |
| 52                                                 | 向左滑动 (Swipe Left)                         |
| 53                                                 | 向右滑动 (Swipe Right)                        |
| 54                                                 | 向上滑动 (Swipe Up)                           |
| 55                                                 | 向下滑动 (Swipe Down)                         |

Wi-Fi收发器： 我们使用一台带有Intel 5300无线网卡的Thinkpad X201笔记本电脑作为Wi-Fi发射器，并使用另外三套作为Wi-Fi接收器。笔记本电脑被放置在矩形区域的四个角上，如图2所示，这一布局灵感来自Widar3.0 [76]中的放置策略。这种布置创造了一个更大的矩形传感区域。它们的高度设置为1.2米，基于ARIL [55]的观察，这被认为对识别全身动作有效。发射器被设置为通过一根天线，在通道128（5.64GHz）上以高通量（IEEE 802.11n）比特率每秒广播200个数据包。每个接收器使用三根天线监控这个通道，因此总共有9条Wi-Fi链路。我们在收发器中安装了一个Wi-Fi工具[19]来进行信道估计并获得30个正交频分复用（OFDM）子载波的信道状态信息（CSI），从而得到尺寸为 (200t)×1×3×3×30 的CSI记录，其中 t 是记录时间的秒数。
