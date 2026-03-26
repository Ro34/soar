import json
import time
import requests
import numpy as np
import concurrent.futures

URL = "http://127.0.0.1:31111/generate"

# ==========================================
# 1. 严格按照官方公布的概率分布生成测试集
# ==========================================
INPUT_BINS = [(0, 4000), (4000, 16000), (16000, 32000), (32000, 128000), (128000, 160000)]
INPUT_PROBS = [0.25, 0.10, 0.15, 0.35, 0.15]
OUTPUT_BINS = [(10, 512), (512, 2048), (2048, 4000), (4000, 16000), (16000, 32000)]
OUTPUT_PROBS = [0.35, 0.25, 0.10, 0.15, 0.15]

# 为了在本地快速拿到基线，我们先测 30 条请求的微缩版
NUM_REQUESTS = 30 

print("🎯 [1/3] 正在生成官方分布的测试集...")
dataset = []
in_choices = np.random.choice(len(INPUT_BINS), size=NUM_REQUESTS, p=INPUT_PROBS)
out_choices = np.random.choice(len(OUTPUT_BINS), size=NUM_REQUESTS, p=OUTPUT_PROBS)

for i in range(NUM_REQUESTS):
    in_b, out_b = INPUT_BINS[in_choices[i]], OUTPUT_BINS[out_choices[i]]
    dataset.append({
        "req_id": i,
        "prompt_len": int(np.random.uniform(max(10, in_b[0]), in_b[1])),
        # 考虑到未量化原模型生成极其缓慢，为了节约你的调试时间，我们暂时把最大生成长度压制到 20。
        # 我们的核心目的是测试并发吞吐和长文本 Prefill 的墙钟时间！
        "gen_len": min(20, int(np.random.uniform(max(1, out_b[0]), out_b[1]))) 
    })

# ==========================================
# 2. 核心压测引擎
# ==========================================
def send_req(req):
    # 用重复单词凑够对应的 Token 长度
    prompt = "A " * req["prompt_len"]
    payload = {"text": prompt, "sampling_params": {"max_new_tokens": req["gen_len"], "temperature": 0.0}}
    try:
        res = requests.post(URL, json=payload, timeout=1200) # 给足 20 分钟超时
        res.raise_for_status()
        return True
    except Exception as e:
        print(f"\n❌ 请求失败 (长度 {req['prompt_len']}): {str(e)[:100]}")
        return False

def run_benchmark(concurrency, desc):
    print(f"\n🚀 开始评测 [{desc}] - 并发度: {concurrency}")
    start_time = time.time()
    
    # 使用线程池发起真实并发
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(send_req, dataset))
        
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r)
    
    print(f"✅ [{desc}] 耗时 (Duration): {total_time:.2f} 秒 | 成功率: {success_count}/{NUM_REQUESTS}")
    return total_time

# ==========================================
# 3. 执行官方规定的三档测试
# ==========================================
print("\n🔥 [2/3] 开始执行三档压力测试 (请确保后台 SGLang 正常运行)")

s1_duration = run_benchmark(1, "S1: 无并发")
s8_duration = run_benchmark(8, "S8: 低并发")
sinf_duration = run_benchmark(NUM_REQUESTS, "S_inf: 高并发满载")

print("\n" + "="*50)
print(f"🏆 你的 A100 (模拟48G) 原精度基线成绩单:")
print(f"   S1    Duration : {s1_duration:.2f} 秒 (占比 40%)")
print(f"   S8    Duration : {s8_duration:.2f} 秒 (占比 30%)")
print(f"   S_inf Duration : {sinf_duration:.2f} 秒 (占比 30%)")
print(f"   正确性系数 C   : 1.0 (未量化满分)")
print("="*50)
print("💡 记住这些时间！接下来的任何算子优化或量化，只要时间比这个短，就是有效提分！")