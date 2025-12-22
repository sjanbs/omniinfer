import pytest
import os
import subprocess
import time
import json
from pathlib import Path
from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock import strart_vllm_mock, cleanup_subprocess
import port_manager
import requests
import concurrent.futures
import random

# Configuration
PREFILL_NUM = 4
DECODE_NUM = 4
proxy_port = 7000
prefill_port_list = None
decode_port_list = None
CUR_DIR = Path(__file__).parent

@pytest.fixture(scope="module")
def setup_teardown():
    global proxy_port
    global prefill_port_list
    global decode_port_list

    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        prefill_port_list = ports["prefill"]
        decode_port_list = ports["decode"]
        print(f"\n[DEBUG] Skipping setup/teardown, {proxy_port=}, {prefill_port_list=}, {decode_port_list=}")
        yield
        return

    ports = port_manager.load_ports(PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    ret = setup_proxy(proxy_port, prefill_port_list, decode_port_list)
    if not ret == 0:
        pytest.fail(f"Start proxy fail")

    processes = strart_vllm_mock(PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail(f"Start vllm fail")

    yield

    teardown_proxy()
    print(f"\n[TEARDOWN] Shutting down {PREFILL_NUM + DECODE_NUM} instances...")
    cleanup_subprocess(processes)

def fetch_post(url, headers, data):
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)  
        return {
            "url": url,
            "status": response.status_code,
            "text": response.text[:200] + "..." if len(response.text) > 200 else response.text
        }
    except requests.exceptions.RequestException as e:
        return {
            "url": url,
            "error": str(e)
        }

def test_chat_completions_with_proxy(setup_teardown):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"  


    data = [
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:1055
            "messages": [{"role": "user", "content": "奔向旁边树丛，拣了一株细长小树，用断剑齐根斩断，削去枝叶，俨然是一根杆棒。皮清玄依样削棒。二道左右夹攻，挺棒向黑驴刺去。那少女轻叱：不要脸！挥刀挡开双棒，就这么一分心，那姓韩乞丐的链子锤与申志凡的长剑前后齐到。那少女急使险招，低头横身，铁锤夹着一股劲风从她脸上掠过。当的一声，弯刀与长剑相交，就在此时，黑驴负痛长嘶，前足提起，原来已让姬清虚刺中了一棒。那姓陈乞丐就地打滚，展开地堂刀法，刀背在驴腿上重重一击，黑驴登时跪倒。这么一来，那少女再也不能乘驴而战，眼见剑锤齐至，当即飞身而起，左手抓住皮清玄的杆棒，用力一拗，杆棒断成两截。她双足着地，回刀横削，格开那姓陈乞丐砍来的一刀。杨过一惊：怎么？她已受了伤？原来那少女左足微跛，纵跃之间显得不甚方便，一直不肯下驴，自是为了这个缘故。杨过侠义之心顿起，待要插手相助，转念却想：我和姑姑好端端在古墓中长相厮守，都是李莫愁那恶女人到来，才闹到这步田地。这女子又冒充我姑姑，要人叫她‘白衣美貌女子’，好不要脸！转过了头，不去瞧她。耳听得兵刃相交叮当不绝，好奇心终于按捺不住，又回过头来，见相斗情势已变，那少女东闪西避，已遮拦多还手少。突然那姓韩乞丐铁锤飞去，那少女侧头让过，正好申志凡长剑削到，玎的一声轻响，将她束发的银环削断了一根，半边鬓发便披垂下来。那少女秀眉微扬，嘴唇一动，脸上登如罩了一层严霜，反手还了一刀。杨过见她扬眉动唇的怒色，心中剧烈一震：姑姑恼我之时，也是这般神色。只因那少女这一发怒，杨过立时决心相助，拾起七八块小石子放入怀中，但见她左支右绌，神情已颇狼狈。申志凡叫道：你跟赤练仙子李莫愁到底怎生称呼？再不实说，可莫怪我们不客气了！那少女弯刀横回，突从他后脑钩了过来。申志凡没料到她会忽施突袭，挡架不及。姓陈乞丐急叫：留神！姬清虚猛力举杆棒向弯刀刃上击去，才救了申志凡性命。五人见她招数如此毒辣，下手加狠。霎时之间，那少女连遇险招。申志凡料想这少女与李莫愁必有渊源，杀伤了她，祸患无穷，反正全真派与李莫愁在山西早动过手，也不怕师伯们怪罪，眼见她并无后援，正好杀了灭口，于是招招指向她要害。杨过见她危在顷刻，再也延缓不得，牵过牛头对住六人，翻身上了牛背，随即溜到牛腹之下，双足勾住牛背，伸指在牛臀上一戳。那牯牛放开四蹄，向六人直冲过去。六人恶斗正酣，突见疯牛冲来，都吃了一惊，四下纵开避让。杨过伏在牛腹之下，看准了五个男子的背心穴道，小石子一枚枚掷出，或中魂门，或中神堂，但听得呛啷、拍喇、哎唷连响，五人双臂酸麻，手中兵刃纷纷落地。杨过却已驱赶牯牛回上山坡。他从牛腹下翻身落地，大叫大嚷：啊哟，大牯牛发疯啦，这可不得了啦！申志凡穴道遭点，兵刃脱手，又不见敌人出手，自料是那少女的帮手所为，此人武功如此高明，那里还敢恋战？幸好双腿仍能迈步，发足便奔，总算他尚有义气，叫道：陈大哥，韩兄弟，咱们走罢！余人不暇细想，也都跟着逃走。皮清玄慌慌张张，不辨东西，反而向那少女奔去。姬清虚大叫：皮师弟，到这里来！皮清玄待要转身，那少女抢上一步，弯刀斫落。皮清玄大惊，手中又没兵刃，忙偏身闪避，那少女弯刀斫出时似东实西，如上却下，冷光闪处，已砍到了他面门。皮清玄危急中举手挡格，嚓的一声，弯刀已削去了他三根手指。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:232
            "messages": [{"role": "user", "content": "杨过安慰道：你爹爹新婚后心中高兴，定是待你更加好些。绿萼摇头道：我宁可他待我更凶些，也别娶新妈妈。杨过父母早死，对这般心情不大了然，有意要逗她开心，道：你新妈妈一定没你一半美。绿萼忙道：你偏说错了，我这新妈妈才真正是美人儿呢。爹爹可为她……为她……昨儿我们把那姓周的老头儿捉了来，若不是爹爹忙着安排婚事，决不会再让这老顽童逃走。杨过又惊又喜，问道：老顽童又逃走了？绿萼秀眉微蹙，道：可不是吗？杨过早料到以周伯通的本事，绝情谷中四弟子纵有渔网，也决拿他不住。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:595
            "messages": [{"role": "user", "content": "黄蓉道：回头我告知她便是，你爷儿俩去敌营走一趟，半天即回，又不是什么大事。杨过心想与黄蓉斗智，处处落于下风，但郭靖诚朴老实，决不是自己对手，同去蒙古军中后对付了他，再回来与小龙女会合不迟，于是略一结束，随同郭靖出城。郭靖骑的是汗血宝马，杨过乘了黄毛瘦马，两匹马脚力均快，不到半个时辰，已抵达蒙古大营。忽必烈听报郭靖竟然来到，又惊又喜，忙叫请进帐来。郭靖走进大帐，只见一位青年王爷居中而坐，方面大耳，两目深陷，不由得一怔：此人竟与他父亲拖雷一模一样。想起少年时与拖雷情深义重，此时却已阴阳相隔，不禁眼眶一红，险些儿掉下泪来。忽必烈下座相迎，一揖到地，说道：先王在日，时常言及郭靖叔叔英雄大义，小侄仰仰慕无已，日来得睹尊颜，实慰生平之愿。郭靖还了一揖，说道：拖雷安答和我情逾骨肉，我幼时母子俩托庇成吉思汗麾下，极仗令尊照拂。令尊英年，如日方中，不意忽尔谢世，令人思之神伤。说着不禁泪下。忽必烈见他言辞恳挚，动了真情，也不由得伤感，便与潇湘子、尹克西等一一引见，请郭靖上座。杨过侍立在郭靖身后，假装与诸人不识。国师等不知他此番随来是何用意，见他不理睬各人，也均不与他说话。麻光佐却大声道：杨兄……下面一个弟字还未出口，尹克西在他大腿上狠狠捏了一把。麻光佐啊哟一声，叫道：干什么？尹克西转过了头不理。麻光佐不知是谁捏他，口中唠唠叨叨骂人，便忘了与杨过招呼。郭靖坐下后饮了一杯马乳酒，不见武氏兄弟，正要动问，忽必烈已向左右吩咐：快请两位武爷。左右卫士应命而出，推了武敦儒、武修文进帐。两人手足都给用牛筋绳绑得结结实实，双足之间的牛筋长不逾尺，迈不开步子，只能慢慢的挨着过来。二武见到师父，满脸羞惭，叫了一声：师父！都低下了头不敢抬起。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:151
            "messages": [{"role": "user", "content": "现下天下英雄会集于此，人人心怀忠义，咱们须得商量个妙策，使得蒙古鞑子不敢来犯我大宋江山。群雄纷纷起立，你一言我一语，都表赞同。此日来赴英雄宴之人多数都是血性汉子，眼见国事日非，大祸迫在眉睫，早就深自忧心，有人提起此事，忠义豪杰自是如响斯应。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },        
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:324
            "messages": [{"role": "user", "content": "黑衣僧一怔，觉得曾在什么地方和此人会过，又觉得他这眼色瞧得自己极不舒服，当即转头避开，过不片刻，忍不住又去望了他一眼。彭长老笑道：下得好大的雪啊，是不是？黑衣僧道：是，好大的雪。彭长老道：来，咱们去瞧瞧雪景。说着推开了板门。黑衣僧道：好，去瞧瞧雪景。站起身来，和他并肩站在门口。杨过虽隔着板壁，也觉彭长老眼光特异，心中隐隐有不祥之感。彭长老道：你师父说得好，杀人是万万不可的，但你全身劲力充溢，若不和人动手，心里便十分难过，是不是啊？黑衣僧迷迷糊糊的应道：是啊！彭长老道：你不妨发掌击这雪人，打好了，那可没有罪孽。黑衣僧望着雪人，双臂举起，跃跃欲试。这时离二僧到来之时已隔了小半个时辰，瘦丐身上又堆了一层白雪，连得他双眼也皆掩没。老道：你双掌齐发，打这雪人，打啊！打啊！打啊！语音柔和，充满了劝诱之意。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        }
    ]

    start_time = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor: 
        future_to_data = {}
        
        for d in data:
            headers = {
                "Content-Type": "application/json",
                "X-Request-Id": f"{''.join(random.choices('123456789', k=5))}",
            }
            future = executor.submit(fetch_post, url, headers, d)
            
            future_to_data[future] = d
            time.sleep(0.01) 
    
    for future in concurrent.futures.as_completed(future_to_data):
        result = future.result()
        results.append(result)

        print(f"Status: {result['status']}, Preview: {result['text']}")
        assert result['status'] == 200

    end_time = time.time()
    print(f"take: {end_time - start_time:.2f} seconds")
    log_file = f"{CUR_DIR}/nginx_access.log"
    num_logs = 5
    print("\n=== verifying load balance ===")
    try:
        analyze_balance(log_file, num_logs)
    
    except Exception as e:
        print(f"\n=== verifying fail: {e} ===")
        raise
    print("\n=== verifying pass ===")

def analyze_balance(log_file, num_logs=5):
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"log {log_file} does not exist")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise RuntimeError(f"failed to read log: {e}")
    
    if not logs:
        raise ValueError("empty log")
    parsed_logs = []
    for line in logs:
        try:
            data = parse_log_line(line) 
            if not data:
                continue
            parsed_logs.append({
                'prefill_idx': int(data['prefill_idx']),
                'decode_idx': int(data['decode_idx']),
                'promt_tks': int(data['promt_tks']),
                'decoded_tks': int(data['decoded_tks']),         
            })
        except KeyError as e:
            print(f"lack of description: {e} (origin log: {line[:100]}...)")

    if not parsed_logs:
        raise ValueError("could not find log")
    
    recent_logs = parsed_logs[-num_logs:]
    prefill_mapping = {232:1, 324:3,1055:0,151:3,595:2}
    decode_counts = {0:0,1:0,2:0,3:0}
    prompt_decode_min_idx = 3
    for req in recent_logs:
        # check whether idx equal to expected idx depends on prompt_tks
        assert prefill_mapping[req['promt_tks']] == req['prefill_idx']
        if req['promt_tks'] == 151:
            prompt_decode_min_idx = req['decode_idx']
        decode_counts[req['decode_idx']] += 1

    assert decode_counts[prompt_decode_min_idx] == 2 


def parse_log_line(line):

    line = line.strip()
    if not line.startswith("{") or not line.endswith("}"):
        return None  

    line = line[1:-1]  
    parts = []
    current = ""
    in_string = False 

    for char in line:
        if char == '"' and not in_string:
            in_string = True 
        elif char == '"' and in_string:
            in_string = False 
        elif char == "," and not in_string:
            parts.append(current)
            current = ""
        else:
            current += char

    if current:
        parts.append(current)

    result = {}
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.replace(".", "", 1).isdigit():
            value = float(value) if "." in value else int(value)
        result[key] = value

    return result
