from PIL import Image
import numpy as np
from fer import FER
from google.colab import files

# 上传图片
uploaded = files.upload()

# 获取上传的图片路径
img_path = list(uploaded.keys())[0]

# 使用PIL打开上传的图片
img = Image.open(img_path)

# 转换为numpy数组
img_np = np.array(img)

# 显示图片信息
print(f"Image size: {img.size}")
img.show()

# 初始化情感分析器
detector = FER()

# 获取所有情绪的置信度
emotions = detector.detect_emotions(img_np)

# 定义情绪与VA值的映射
emotion_va_map = {
    "happy": {"valence": 0.8, "arousal": 0.6},
    "sad": {"valence": -0.6, "arousal": -0.5},
    "angry": {"valence": -0.8, "arousal": 0.9},
    "surprise": {"valence": 0.5, "arousal": 0.8},
    "disgust": {"valence": -0.7, "arousal": -0.4},
    "fear": {"valence": -0.6, "arousal": 0.7},
    "neutral": {"valence": 0.0, "arousal": 0.0}
}

# 初始化 Valence 和 Arousal 的总值
total_valence = 0
total_arousal = 0
total_confidence = 0

# 确保每个情绪都正确映射并按置信度比例加权计算VA值
if emotions:  # 确保emotions列表不为空
    emotion_data = emotions[0]["emotions"]
    for emotion, confidence in emotion_data.items():
        if emotion.lower() in emotion_va_map:
            va_values = emotion_va_map[emotion.lower()]
            
            # 按照置信度比例累加 VA 值
            total_valence += va_values["valence"] * confidence
            total_arousal += va_values["arousal"] * confidence
            total_confidence += confidence
            
            # 输出每个情绪和其占比
            print(f"{emotion}: Confidence = {confidence:.2f}, Valence = {va_values['valence']}, Arousal = {va_values['arousal']}")

    # 计算加权平均值
    if total_confidence > 0:
        avg_valence = total_valence / total_confidence
        avg_arousal = total_arousal / total_confidence
        print(f"\nAverage VA values: Valence = {avg_valence:.2f}, Arousal = {avg_arousal:.2f}")
    else:
        print("No emotions detected.")
else:
    print("No emotions detected.")
