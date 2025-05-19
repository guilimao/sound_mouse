import sounddevice as sd
import numpy as np
from scipy.fft import rfft, rfftfreq
from collections import deque
import time
import pyautogui


# 参数配置
fs = 44100
block_size = 4096
duration = 30
smooth_window = 5

screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

# 初始化变量
freq_history = deque(maxlen=smooth_window)
max_freq = 0
max_db = 0

def audio_callback(indata, frames, time, status):
    global max_freq, max_db
    
    signal = indata[:, 0] - np.mean(indata[:, 0])
    window = np.hanning(len(signal))
    signal_windowed = signal * window
    
    # 分贝计算
    rms = np.sqrt(np.mean(signal**2))
    db = 20 * np.log10(rms / 20e-6) if rms > 0 else -np.inf
    
    # FFT分析
    yf = np.abs(rfft(signal_windowed))
    xf = rfftfreq(len(signal_windowed), 1/fs)
    
    # 峰值检测（带阈值）
    min_freq_idx = np.argmax(xf > 50)
    yf_segment = yf[min_freq_idx:]
    threshold = np.max(yf_segment) * 0.2 if len(yf_segment) > 0 else 0
    significant_peaks = np.where(yf_segment > threshold)[0]
    
    if len(significant_peaks) > 0:
        peak_idx = significant_peaks[np.argmax(yf_segment[significant_peaks])]
        current_freq = xf[min_freq_idx + peak_idx]
    else:
        current_freq = 0
    
    # 滑动平均
    freq_history.append(current_freq)
    smoothed_freq = np.mean(freq_history)
    
    max_freq = smoothed_freq
    max_db = db


def map_value(old_val, old_min, old_max, new_min, new_max, clamp=True):
    """
    将值从原范围线性映射到新范围
    :param clamp: 是否限制结果在目标范围内
    """
    # 计算比例
    normalized = (old_val - old_min) / (old_max - old_min)
    # 映射到新范围
    new_val = normalized * (new_max - new_min) + new_min
    # 可选：限制在目标范围内
    if clamp:
        new_val = max(min(new_val, new_max), new_min)
    return new_val

def map_to_screen(prog_x, prog_y):
    # 获取当前屏幕尺寸
    screen_width, screen_height = pyautogui.size()
    
    # 映射 X 轴：原范围 [100, 200] → 屏幕 [0, screen_width]
    screen_x = map_value(
        prog_x,
        old_min=100, old_max=200,
        new_min=0, new_max=screen_width
    )
    
    # 映射 Y 轴：原范围 [-80, -50] → 屏幕 [0, screen_height]
    # （若需要反转方向：new_min=screen_height, new_max=0）
    screen_y = map_value(
        prog_y,
        old_min=80, old_max=70,
        new_min=0, new_max=screen_height
    )
    
    return (int(screen_x), int(screen_y))

# 启动音频流
with sd.InputStream(callback=audio_callback, samplerate=fs, channels=1, blocksize=block_size):
    print("实时频率检测运行中...（按Ctrl+C停止）")
    start_time = time.time()
    try:
        while time.time() - start_time < duration:

            target_x,target_y = map_to_screen(max_freq,max_db)
            pyautogui.moveTo(target_x,target_y,duration=0.02)
            print(f"\r主频率: {max_freq:.1f} Hz | 响度: {max_db:.1f} dB", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print("\n结束")