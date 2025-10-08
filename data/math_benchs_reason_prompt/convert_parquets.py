import pandas as pd

INSTRUCTION = "High Reasoning Effort: You have unlimited time to think and respond to the user\u2019s question. There is no need to worry about reasoning time or associated costs. Your only goal is to arrive at a reliable, correct final answer. Feel free to explore the problem from multiple angles, and try various methods in your reasoning. This includes reflecting on reasoning by trying different approaches, verifying steps from different aspects, and rethinking your conclusions as needed. You are encouraged to take the time to analyze the problem thoroughly, reflect on your reasoning promptly and test all possible solutions. Only after a deep, comprehensive thought process should you provide the final answer, ensuring it is correct and well-supported by your reasoning.\nIf it's a math problem, please reason step by step, and put your final answer within \\boxed{}."

for data_type in ["aime", "amc", "math",  "minerva",  "olympiad_bench"]:
    df = pd.read_parquet(f"/volume/pt-train/users/syguo/verl_trpo/data/math_benchs/{data_type}.parquet")
    # {'content': "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop. Let's think step by step and output the final answer within \\boxed{}."}
    for i in range(len(df)):
        df.iloc[i]['prompt'][0]['content'] = df.iloc[i]['prompt'][0]['content'].replace("Let's think step by step and output the final answer within \\boxed{}.", "").rstrip()
        df.iloc[i]['prompt'][0]['content']  = INSTRUCTION + "\n\n" + df.iloc[i]['prompt'][0]['content']
        
    # print sample propmt
    print(f"Sample prompt for {data_type}: {df.iloc[-1]['prompt']}")
    df.to_parquet(f"/volume/pt-train/users/syguo/verl_trpo/data/math_benchs_reason_prompt/{data_type}.parquet", index=False)