import requests
import time

URL = "http://127.0.0.1:31111/generate"

# 构造 12 万 Token 的超级长文本（模拟官方 128K-160K 区间）
# "hello " 大约算 1 个 token，我们直接循环 120,000 次
prompt_length = 120000
print(f"🚧 正在构造约 {prompt_length} Tokens 的超级 Prompt...")
prompt = "hello " * prompt_length

payload = {
    "text": prompt,
    "sampling_params": {
        "max_new_tokens": 10,  # 只要能吐出 10 个字，就说明最艰难的 Prefill 阶段挺过去了！
        "temperature": 0.0
    }
}

print("🚀 发射！120K 超长文本已送达 SGLang！")
print("👀 警告：请立即切到 nvidia-smi 窗口，观察 GPU 2 的显存是否会瞬间爆炸！")

start_time = time.time()

try:
    # 设定 10 分钟超时，因为 120K 的无量化 Prefill 可能会非常非常慢
    response = requests.post(URL, json=payload, timeout=600) 
    response.raise_for_status()
    end_time = time.time()
    result = response.json()
    
    print("\n🎉 奇迹发生！GPU 居然活下来了！")
    print(f"⏱️ 首字+生成总耗时: {end_time - start_time:.2f} 秒")
    print(f"📝 模型回复: {result.get('text', '').strip()}")
    
except requests.exceptions.RequestException as e:
    print("\n💀 悲剧了！服务大概率已经 OOM 崩溃，或者显存撑爆导致底层报错被 Kill 了。")
    print(f"❌ 错误现场: {e}")