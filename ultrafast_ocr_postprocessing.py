# -*- coding: utf-8 -*-
"""
文字解码后处理模块
"""

import os
import numpy as np
from typing import Tuple, List


class TextDecoder:
    """文字解码器"""
    
    def __init__(self, dict_path: str):
        """
        初始化解码器
        
        Args:
            dict_path: 字符字典路径
        """
        self.character_dict = self.load_character_dict(dict_path)
        self.character_dict_size = len(self.character_dict)
    
    def load_character_dict(self, dict_path: str) -> List[str]:
        """
        加载字符字典
        
        Args:
            dict_path: 字典文件路径
            
        Returns:
            字符列表
        """
        character = []
        
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        char = line.strip()
                        if char:
                            character.append(char)
                            
                print(f"✅ 加载字符字典: {len(character)} 个字符")
                
            except Exception as e:
                print(f"⚠️ 加载字典失败: {e}, 使用默认字典")
                character = self.get_default_character_dict()
        else:
            print(f"⚠️ 字典文件不存在: {dict_path}, 使用默认字典")
            character = self.get_default_character_dict()
        
        # 添加特殊字符
        character = ["<blank>"] + character + [" "]
        
        return character
    
    def get_default_character_dict(self) -> List[str]:
        """获取默认字符字典"""
        # 基础字符集
        chars = []
        
        # 数字
        chars.extend(list("0123456789"))
        
        # 英文字母
        chars.extend(list("abcdefghijklmnopqrstuvwxyz"))
        chars.extend(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        
        # 常用符号
        chars.extend(list(".,!?:;\"'()[]{}+-=*/%@#$&"))
        
        # 常用中文字符（简化版）
        common_chinese = (
            "的一是在了不和有大这主中人上为们地个用工时要动国产以我到他会作来分生对于学下级就年阶"
            "义发成部民可出能方进同行面说种过命度革而多子后自社加小机也经力线本电高量长党得"
            "实家定深法表着水理化争现所二起政三好十战无农使性前等反体合斗路图把结第里正新开"
            "论之物从当两些还天资事队批如应形想制心样干都向变关点育重其思与间内去因件日利相"
            "由压员气业代全组数果期导平各基或月毛然问比展那它最及外没看治提五解系林者米群头"
            "意只明四道马认次文通但条较克又公孔领军流入接席位情运器并飞原油放立题质指建区验"
            "活众很教决特此常石强极土少已根共直团统式转别造切九你取西持总料连任志观调七么山"
            "程百报更见必真保热委手改管处己将修支识病象几先老光专什六型具示复安带每东增则完"
            "风回南广劳轮科北打积车计给节做务被整联步类集号列温装即毫知轴研单色坚据速防史拉"
            "世设达尔场织历花受求传口断况采精金界品判参层止边清至万确究书术状厂须离再目海交"
            "权且儿青才证低越际八试规斯近注办布门铁需走议县兵固除般引齿千胜细影济白格效置推"
            "空配刀叶率述今选养德话查差半敌始片施响收华觉备名红续均药标记难存测士身紧液派准"
            "斤角降维板许破述技消底床田势端感往神便贺村构照容非搞亚磨族火段算适讲按值美态黄"
            "易彪服早班麦削信排台声该击素张密害侯草何树肥继右属市严径螺检左页抗苏显苦英快称"
            "坏移约巴材省黑武培著河帝仅针怎植京助升王眼她抓含苗副杂普谈围食射源例致酸旧却充"
            "足短划剂宣环落首尺波承粉践府鱼随考刻靠够满夫失包住促枝局菌杆周护岩师举曲春元超"
            "负砂封换太模贫减阳扬江析亩木言球朝医校古呢稻宋听唯输滑站另卫字鼓刚写刘微略范供"
            "阿块某功套友限项余倒卷创律雨让骨远帮初皮播优占死毒圈伟季训控激找叫云互跟裂粮粒"
            "母练塞钢顶策双留误础吸阻故寸盾晚丝女散焊功株亲院冷彻弹错散商视艺灭版烈零室轻血"
            "倍缺厘泵察绝富城冲喷壤简否柱李望盘磁雄似困巩益洲脱投送奴侧润盖挥距触星松送获兴"
            "独官混纪依未突架宽冬章湿偏纹吃执阀矿寨责熟稳夺硬价努翻奇甲预职评读背协损棉侵灰"
            "虽矛厚罗泥辟告卵箱掌氧恩爱停曾溶营终纲孟钱待尽俄缩沙退陈讨奋械载胞幼哪剥迫旋征"
            "槽倒握担仍呀鲜吧卡粗介钻逐弱脚怕盐末阴丰雾冠丙街莱贝辐肠付吉渗瑞惊顿挤秒悬姆烂"
            "森糖圣凹陶词迟蚕亿矩康遵牧遭幅园腔订香肉弟屋敏恢忘编印蜂急拿扩伤飞露核缘游振操"
        )
        
        chars.extend(list(common_chinese))
        
        return chars
    
    def decode_recognition(self, pred: np.ndarray) -> Tuple[str, float]:
        """
        解码识别结果(CTC解码)
        
        Args:
            pred: 模型预测结果 [1, seq_len, num_classes] 或 [seq_len, num_classes]
            
        Returns:
            (识别的文字, 平均置信度)
        """
        try:
            # 处理输入维度
            if len(pred.shape) == 3:
                pred = pred[0]  # 去掉batch维度
            elif len(pred.shape) != 2:
                return "", 0.0
            
            # 获取预测序列
            pred_indices = np.argmax(pred, axis=1)
            pred_scores = np.max(pred, axis=1)
            
            # CTC解码
            decoded_chars = []
            decoded_scores = []
            
            last_idx = -1
            for i, (idx, score) in enumerate(zip(pred_indices, pred_scores)):
                # 跳过blank字符(索引0)
                if idx == 0:
                    last_idx = idx
                    continue
                
                # CTC规则：跳过连续重复的字符
                if idx == last_idx:
                    continue
                
                # 检查索引是否有效
                if idx < len(self.character_dict):
                    char = self.character_dict[idx]
                    decoded_chars.append(char)
                    decoded_scores.append(score)
                
                last_idx = idx
            
            # 组合结果
            text = ''.join(decoded_chars)
            
            # 计算平均置信度
            if decoded_scores:
                confidence = float(np.mean(decoded_scores))
            else:
                confidence = 0.0
            
            return text, confidence
            
        except Exception as e:
            print(f"⚠️ 文字解码失败: {e}")
            return "", 0.0
    
    def beam_search_decode(self, 
                          pred: np.ndarray, 
                          beam_width: int = 5) -> List[Tuple[str, float]]:
        """
        束搜索解码(更准确但更慢)
        
        Args:
            pred: 模型预测结果
            beam_width: 束宽度
            
        Returns:
            [(候选文字, 置信度), ...] 按置信度排序
        """
        if len(pred.shape) == 3:
            pred = pred[0]
        
        # 简化版束搜索实现
        seq_len, num_classes = pred.shape
        
        # 初始化束
        beams = [('', 0.0, -1)]  # (text, log_prob, last_idx)
        
        for t in range(seq_len):
            new_beams = []
            
            for text, log_prob, last_idx in beams:
                # 获取当前时刻的概率分布
                probs = pred[t]
                
                # 取top-k候选
                top_k = min(beam_width, len(probs))
                top_indices = np.argpartition(probs, -top_k)[-top_k:]
                
                for idx in top_indices:
                    new_log_prob = log_prob + np.log(max(probs[idx], 1e-8))
                    
                    if idx == 0:  # blank
                        new_beams.append((text, new_log_prob, idx))
                    elif idx != last_idx and idx < len(self.character_dict):
                        char = self.character_dict[idx]
                        new_text = text + char
                        new_beams.append((new_text, new_log_prob, idx))
                    else:
                        new_beams.append((text, new_log_prob, idx))
            
            # 保留top beam_width个候选
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # 转换为结果格式
        results = []
        for text, log_prob, _ in beams:
            if text:  # 过滤空字符串
                confidence = np.exp(log_prob / max(1, len(text)))  # 归一化
                results.append((text, float(confidence)))
        
        return results
    
    def post_process_text(self, text: str) -> str:
        """
        文本后处理
        
        Args:
            text: 原始识别文本
            
        Returns:
            处理后的文本
        """
        if not text:
            return text
        
        # 1. 去除首尾空格
        text = text.strip()
        
        # 2. 替换常见错误
        error_corrections = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
            'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
            'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e',
            '，': ',', '。': '.', '？': '?', '！': '!', '：': ':',
            '；': ';', '"': '"', '"': '"', ''': "'", ''': "'",
        }
        
        for old, new in error_corrections.items():
            text = text.replace(old, new)
        
        # 3. 合并多个空格
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def calculate_edit_distance(self, s1: str, s2: str) -> int:
        """
        计算编辑距离
        
        Args:
            s1, s2: 两个字符串
            
        Returns:
            编辑距离
        """
        m, n = len(s1), len(s2)
        
        # 创建dp表
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 填充dp表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,    # 删除
                        dp[i][j - 1] + 1,    # 插入
                        dp[i - 1][j - 1] + 1  # 替换
                    )
        
        return dp[m][n]
