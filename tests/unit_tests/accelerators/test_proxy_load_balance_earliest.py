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
            "max_tokens": 20,#prompt length: 170
            "messages": [{"role": "user", "content": "他们约定在山下普光寺中聚会，以手击碑石为号。你无意中在碑上拍了一下，又显出功力惊人，无怪我那些没用的徒子徒孙便大惊小怪。那两个大魔头都是蒙古密教弟子，武功不弱，今年到中原几下出手，震动武林。你在桃花岛隐居，因而不知。那贵公子是蒙古的王子，据说还是大汗成吉思汗的近系子孙，旁人都叫他作霍都王子。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },                
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:332
            "messages": [{"role": "user", "content": "洪七公听他这么说，连连点头，道：好小子，原来他是你义父。那知欧阳锋突然跃起，叫道：老叫化，咱们拳脚比不出胜败，再比兵器。洪七公听他叫自己老叫化，微微一笑，摇头道：不比啦，算你胜就是。欧阳锋道：什么算不算的？我非杀了你不可。回手折了根树枝，拉去枝叶，成为一条棍棒，向洪七公兜头击落。他的蛇杖当年纵横天下，厉害无比，现下杖头虽然无蛇，但这一杖击将下来，杖头未至，烈风已将杨过逼得难以喘气。杨过忙跃开躲避，看洪七公时，只见他拾起地下一根树枝，当作短棒，二人又已斗在一起。洪七公的打狗棒法世间无双，但轻易不肯施展，除此之外尚有不少精妙棒法，此时便逐一使将出来。这场拚斗，与适才比拚拳脚又是另一番光景，但见杖去灵蛇盘舞，棒来神龙夭矫，或似长虹经天，或若流星追月，只把杨过瞧得惊心动魄，如醉如痴。二人杖去棒来，直斗到傍晚，兀自难分胜败。请基于以上故事情节续写小说，字数不少于1200字。"}],
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
            "max_tokens": 20,#prompt length: 913
            "messages": [{"role": "user", "content": "常言道明枪易躲，暗箭难防。忽必烈道：适才攻城之时，你站在我身旁，只怕他在城头已然瞧见。杨过道：小人已防到此着，攻城之时，与龙姑娘均以大帽遮眉、皮裘围颈，他决计认不出来。忽必烈道：既是如此，盼你立此大功，封赏之约，决不食言。杨过随口道谢一声，正要转身与小龙女一齐辞出，却见金轮国师、潇湘子、尹克西诸人脸上均有异色，心念一动：这些人均怕我此去刺死郭靖，得了蒙古第一勇士的封号，定要从中阻挠。向忽必烈道：王爷，小人去刺郭靖，乃是为报私仇，兼之要以他的首级去换救命丹药，如能托王爷之福，大事得成，那蒙古第一勇士的封号却万万不敢领受。忽必烈问道：这却为何？杨过道：小人武功远不及在座诸位，如何敢称第一勇士？王爷须得应允此事，小人方敢动身。忽必烈见他言辞诚恳，确是真情，又见旁人神情，已猜到他心意，说道：既是如此，人各有志，我也不便勉强。国师等听忽必烈如此说，果然均有欣慰之色。杨过圈转马头，与小龙女并骑向襄阳驰去，在途中摔去了大帽皮裘，回复汉人打扮，到得城下时天已向晚，见城门紧闭，城头一队队兵卒手执火把，来去巡逻。杨过大声叫道：我姓杨名过，特来拜见郭靖郭大爷。城上守将听得呼声，见他只有一名女子相从，当即向郭靖禀报。过不片时，两个青年走上城头，向下一望，一人叫道：原来是杨大哥，只你们两位吗？杨过见是武氏兄弟，心想：郭靖害我父亲，不知武氏兄弟的父亲曾否在旁相助？说道：武大哥，武二哥，郭伯伯在不在城里？武修文道：在的，杨大哥请进来罢。命兵卒打开城门，放下吊桥，让杨过与小龙女入城。二武引着二人来到一座大屋之前。郭靖满脸堆欢，抢出门来，向小龙女一揖为礼，拉着杨过的手笑道：过儿，你们来得正好。鞑子攻城正急，两位一到，我平添臂助，真乃满城百姓之福。小龙女是杨过之师，郭靖对她以平辈之礼相敬，客客气气的让着进屋，对杨过则十分亲热。杨过左手让他握着，想起此人乃杀父大仇，居然这般假惺惺作态，恨不得拔出剑来立时刺死了他，但忌惮他武功，不敢贸然动手，脸上强露笑容，说道：郭伯伯安好。他满腔愤恨，没跪下磕头。郭靖豁达大度，于此细节也没留心。到得厅上，杨过要入内拜见黄蓉。郭靖笑道：你郭伯母即将临盆，这几天身子不适，日后再见罢。杨过暗喜：黄蓉智计过人，我只担心给她看出破绽，此人抱恙，真是天助我成功。说话之间，中军进来禀道：吕大帅请郭大爷赴宴，庆贺今日大胜鞑子。郭靖道：你回禀大帅，多谢赐宴。我有远客光临，不能奉陪了。中军见杨过年纪甚轻，并无特异之处，不知郭靖何以对他如此看重，为了陪伴这个少年，竟推却元帅的庆功宴，不由得满心奇怪，回去禀知吕文焕。郭靖在内堂自设家常酒宴，为小龙女与杨过接风，由朱子柳、鲁有脚、武氏兄弟、郭芙诸人相陪。朱子柳向杨过连声称谢，说亏得他从霍都取得解药，治了他身上之毒。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
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
            "max_tokens": 20,#prompt length:1064
            "messages": [{"role": "user", "content": "杨过与小龙女互视一眼，均想：我二人若能撇开了旁人，在静室中相处片刻，死亦甘心。当即携手向西，从侧门出去，走过两间房，来到第三间房前。小龙女眼光始终没离开杨过之脸，见房门闭着，也不细看，伸手推开，正要跨过门槛进去，杨过猛地想到一事，忙伸手拉住道：小心了。小龙女道：怎么？杨过左足踏在门槛之外，右足跨过门槛往地板上一点，立即缩回，丝毫不见异状。小龙女道：你怕谷主要暗害咱们吗？他这人很好，决不致于……刚说完这三句话，猛听得嗤嗤声响，眼前白光闪动，八柄利剑自房门上下左右挺出，纵横交错，布满入口，若有人于此时踏步进门，武功再高，也难免给这八柄利剑从四面八方在身上对穿而过。小龙女透了口长气，说道：过儿，这谷主恁地歹毒，我真瞧错他的为人了。咱们也不用跟他比什么剑，这就走罢。忽听身后有人说道：谷主请两位入室拣剑。两人回过头来，见八名绿衫弟子手持带刀渔网，拦在身后，自是谷主防杨龙二人相偕逃走，派人截住后路。小龙女的金铃索已为黑剑割断，再不能如适才这般遥点绿衫弟子的穴道。小龙女向杨过道：你说这室中还有什么古怪？杨过将她双手握在掌中，说道：姑姑，此刻你我相聚，复有何憾？便万剑穿心，你我也死在一起。小龙女心中也是柔情万种。两人一齐步入剑室，杨过随手把门带上。只见室中壁上、桌上、架上、柜中、几间，尽皆列满兵刃，式样繁多，十之八九都是古剑，或长逾七尺，或短仅数寸，有的铁锈斑驳，有的寒光逼人，二人眼光缭乱，一时也看不清这许多。小龙女对杨过凝视半晌，突然嘤的一声，投入他怀中。杨过将她紧紧抱住，在她嘴上亲去。小龙女在他一吻之下，心魂俱醉，双手伸出去搂住他头颈，凑嘴回吻。突然砰的一声，室门推开，一名绿衫弟子厉声说道：谷主有令，拣剑后立即出室，不得逗留。杨过脸上一红，当即双手放开。小龙女却想自己心爱杨过，二人相拥而吻决没什么不该，但既有人在旁干扰，难以畅怀，叹了一口气，轻声说道：过儿，待咱们打败了那谷主，你再这般亲我。杨过笑着点了点头，伸左手搂住她腰，柔声道：我永生永世也亲你不够。你拣兵器罢。小龙女道：这里的兵刃瞧来果然均是异物，没一件不好。咱们古墓里也没这么多。于是先从壁间逐一看去，要想拣一对长短轻重都是一般的利剑，但瞧来瞧去，各剑均自不同。她一面看，一面问道：适才进室之时，你怎知此处装有机关？杨过道：我从谷主的脸色和眼光中猜想而知。他本想娶你为妻，但听到你要和我联手斗他，便想杀你了。以他为人，我不信他会好心让咱们来拣选兵刃。小龙女又低低叹了口气，道：咱们使玉女素心剑法，能胜得了他么？杨过道：他武功虽强，却也并不在金轮国师之上。我二人联手胜得国师，谅来也可胜他。小龙女道：是了，国师不住激他和我二人动手，他是要瞧个清楚。杨过微笑道：人心鬼蜮，你也领会得一些了。我只担心你身子，刚才你又呕了血。小龙女笑靥如花，道：你知道的，我伤心气恼的时候才会呕血，现下我欢喜得很，这点内伤不算什么。你也呕了血，不打紧罢？杨过道：我见了你，什么都不碍事了。小龙女柔声道：我也这样。顿了一顿，又道：你近来武功大有进境，合斗国师之时咱们尚且能胜，何况今日？杨过听了此言，也觉这场比试定能取胜，握着她手说道：我想要你答允一件事，不知你肯不肯？小龙女柔声道：你又何必问我？我早已不是你师父，是你妻子啦。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length: 1199
            "messages": [{"role": "user", "content": "霍都给他打断话头，大是气恼，喝道：小畜生！快滚开！杨过叫道：小畜生骂谁？霍都听他问小畜生骂谁，顺口答道：小畜生骂你！他怎知南方孩子向来以这般套子斗口，一不留神，已自上当。杨过哈哈大笑，说道：不错，正是小畜生骂我！大厅上情势本来甚为紧张，却给这少年突然这么一个打岔，群雄都笑了出来。霍都大怒，折扇直出，往杨过头顶击落。群雄适才均见霍都武功了得，这一扇如打在杨过头上，不死也必重伤，齐声呼叫：住手！不得以大欺小。郭靖飞身抢出，正要伸手夺扇，杨过头一低，已从霍都手臂下钻过，桨柄回绕，使出打狗棒法的缠字诀，在霍都脚下一绊。霍都立足不稳，一个踉跄，险些跌倒，总算他武功高强，将跌势硬生生变为跃势，凌空窜起，再稳稳落下。郭靖一怔，问道：过儿，怎么了？杨过笑道：没什么。这厮瞧不起洪老帮主的打狗棒法，我就想用打狗棒法摔他个筋斗，可惜给他逃开了。郭靖大奇，又问：你怎么会使？杨过撒谎道：适才鲁帮主和他动手，我瞧了之后，学得几招。郭靖自己天资鲁钝，只道世上聪明之人甚多，对他的话倒也信了八九成。霍都这么一绊，料得是自己不小心，怎想得到这个少年竟有高明武功，心想眼下争盟主是大事，办完正事再打发这小子不迟，大踏步走到郭靖面前，朗声道：郭大侠，今日比武是我们胜了，我师金轮国师是天下武林盟主。可有那一位不服……他说未说完，杨过悄悄走到他身后，桨柄疾送，使出打狗棒法中第四招戳字诀，忽地向他臀上戳去。以霍都的武功修为，背后有人突施暗算，岂有不知之理？可是一来他没将杨过放在眼里，二来打狗棒法端的神奇奥妙，他虽惊觉，急闪之际终究还是差了这么几寸，噗的一下，正中臀部。饶是他内功深厚，臀部又是多肉之处，这一下却也甚为疼痛，兼之出其不意，他只道定可避过，偏偏竟又戳中，不由得啊的一声叫了出来。喝道：什么东西？我就不服！霎时之间，厅上笑声大作。群雄都想这少年不但顽皮，兼且大胆，这蒙古王子居然两次着了他道儿。至此地步，霍都焉得不恼？反手一掌，要先打他个耳光，出了口恶气再说。他虽只顺手一掌，但掌力含劲蓄势，实是蒙古金刚宗武功的精要，预拟一掌要将这少年打昏躺下。郭靖知道厉害，左手探出，反手一勾，已将他手掌抓住，劝道：阁下怎能跟小孩儿一般见识？霍都给他一把抓住，但感半身发麻，不禁惊怒交集。杨过乘势横过柄，重重一棍打在他臀上，叫道：小畜生不听话，爸爸打你屁股！郭靖喝道：过儿快退开，不许胡闹！群豪已嘻嘻哈哈的笑成一团。蒙古一边的众武士纷纷叫嚷：两个打一个么？不要脸！这算不算比武？郭靖一怔，放脱了霍都。黄蓉见杨过适才这一绊一戳，确是打狗棒法招数，心下大疑：他从何处偷学得到这路棒法？难道这几个月来我教鲁有脚之时，每天他都来偷看？但我教棒时每次均四下查过，他怎能瞒得过我？叫道：靖哥哥，你来。郭靖回到妻子身旁，但他担心杨过吃亏，眼光仍是不离厅心二人。只见霍都挥掌飞脚，不住向杨过攻去。杨过一面闪避，一面大叫：打你屁股，打你屁股！横桨柄不住向他臀部抽击，此时霍都展开身法，自己打他不着，每一棍都落了空。霍都用折扇想打杨过脑袋，杨过却用铁桨柄去打他后臀，两人你追我赶，在厅上迅速异常的兜圈子，谁也打不着谁。旁观众人初时只觉滑稽古怪，待见二人绕了几个圈子，都惊讶起来。杨过年纪虽小，然脚步轻盈，身手迅捷，轻功似犹胜对手。霍都几次飞步击打，都给他巧妙避开。点苍渔隐与达尔巴本来各执兵刃，怒目对视，一个要冲上去再打，一个全神戒备，以防对方突袭，见霍都竟奈何不了这少年，都感诧异，一个咧开大嘴嘻嘻而笑，一个以蒙语叽哩咕噜的咒骂。转瞬间霍杨二人又绕了三个圈子，霍都已瞧出对方轻身功夫了得，一味跟他追逐，说不定竟还输了，突然转身，急伸左掌迎面去抓他桨柄，右手扇子往他腿侧环跳穴上点去。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        }               
    ]
    start_time = time.time()
    results = []
    POST_DATA_COUNT = 25
    # TODO max_workers_num depends on len(data)
    with concurrent.futures.ThreadPoolExecutor(max_workers = len(data)) as executor: 
        future_to_data = {}
        
        for d_index in range(POST_DATA_COUNT):
            headers = {
                "Content-Type": "application/json",
                "X-Request-Id": f"{''.join(random.choices('123456789', k=5))}",
            }
            future = executor.submit(fetch_post, url, headers, data[d_index % len(data)])
            future_to_data[future] = data[d_index % len(data)]
            time.sleep(0.02) 
    
    for future in concurrent.futures.as_completed(future_to_data):
        result = future.result()
        results.append(result)

        print(f"Status: {result['status']}, Preview: {result['text']}")
        assert result['status'] == 200

    end_time = time.time()
    print(f"take: {end_time - start_time:.2f} seconds")
    # TODO merge into one file
    log_file = f"{CUR_DIR}/nginx_access.log"
    num_logs = POST_DATA_COUNT
    print("\n=== verifying load balance ===")
    try:
        analyze_balance(log_file, num_logs, len(data))
    
    except Exception as e:
        print(f"\n=== verifying fail: {e} ===")
        raise
    print("\n=== verifying pass ===")

def analyze_balance(log_file, num_logs, num_data):
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
                'time': float(data['time'])
            })
        except KeyError as e:
            print(f"lack of description: {e} (origin log: {line[:100]}...)")

    if not parsed_logs:
        raise ValueError("could not find log")
    
    recent_logs = parsed_logs[-num_logs:]
    recent_logs.sort(key=lambda x: x['time'])
    # pick static mapping for earliest algo
    recent_logs = recent_logs[: num_data]
    '''
    [NOTICE] idx depends on PREFILL_NUM
             probably report wrong in future if any optimization on processing prompt length

    '''
    prefill_mapping = {170:0, 332:1, 595:2, 913:3, 1055:0, 1064:1, 1199: 2}
    decode_counts = {0:0, 1:0, 2:0, 3:0}
    prompt_decode_min_idx = 3
    for req in recent_logs:
        assert prefill_mapping[req['promt_tks']] == req['prefill_idx']
        if req['promt_tks'] == 170:
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