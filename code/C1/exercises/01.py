# LangChain代码最终得到的输出携带了各种参数，查询相关资料尝试把这些参数过滤掉得到content里的具体回答

import re

with open('01.txt', 'r', encoding = 'utf-8') as f:
    contents = f.read()
    # print(contents)
    pattern = r"content='(.*?)' additional_kwargs" 
    match = re.search(pattern, contents, re.DOTALL) # 去掉
    content = match.group(1)
    content = re.sub(r'\*+', '', content) # 去掉**之类的符号
    # print(content)

    lines = [line.strip() for line in content.split('\\n') if line.strip()] # 按照\n来分割成条目
    print(lines)

    '''
    输出结果：
    ['基于提供的上下文，文中列举了以下例子：', '1. 强化学习与监督学习的对比', '监督学习： 图片分类（区分汽车、飞机、椅子）。', 
    '强化学习： 雅达利游戏 Breakout（打砖块游戏）。', '2. 强化学习的现实生活与游戏例子', '自然界： 羚羊（通过试错学习奔跑）。', 
    '金融： 股票交易。', '游戏： 雅达利游戏 Pong（乒乓球）。', '围棋： AlphaGo（击败人类顶尖棋手）。', '3. Gym 库中的具体任务', 
    'MountainCar-v0： 小车上山。', '4. 探索和利用的比喻', '选择餐馆： 去最喜欢的餐馆（利用）vs. 尝试新餐馆（探索）。', 
    '做广告： 采用最优广告策略（利用）vs. 尝试新广告策略（探索）。', '挖油： 在已知地方挖油（利用）vs. 在新地方挖油（探索）。', 
    '玩游戏： 玩《街头霸王》游戏（一直出脚 vs 尝试大招/新招式）。']
    '''