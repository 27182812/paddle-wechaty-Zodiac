
import os
import cv2
import asyncio
import numpy as np
import paddlehub as hub
import json
import urllib.parse
import urllib.request
import match
import chat

from wechaty import (
    Contact,
    FileBox,
    Message,
    Wechaty,
    ScanStatus,
)

os.environ['WECHATY_PUPPET'] = "wechaty-puppet-service"
os.environ['WECHATY_PUPPET_SERVICE_TOKEN'] = "puppet_padlocal_XXX" ## 你自己的token

def chinese_shuxiang(year):
    shuxiang_map = {
        u'鼠': 1900,

        u'牛': 1901,

        u'虎': 1902,

        u'兔': 1903,

        u'龙': 1904,

        u'蛇': 1905,

        u'马': 1906,

        u'羊': 1907,

        u'猴': 1908,

        u'鸡': 1909,

        u'狗': 1910,

        u'猪': 1911}

    for k, v in shuxiang_map.items():

        if (year % v % 12) == 0:
            return k


def xingzuo(month, day):
    xingzuo_map = {
        u'白羊座': [(3, 21), (4, 20)],

        u'金牛座': [(4, 21), (5, 20)],

        u'双子座': [(5, 21), (6, 21)],

        u'巨蟹座': [(6, 22), (7, 22)],

        u'狮子座': [(7, 23), (8, 22)],

        u'处女座': [(8, 23), (9, 22)],

        u'天秤座': [(9, 23), (10, 22)],

        u'天蝎座': [(10, 23), (11, 21)],

        u'射手座': [(11, 23), (12, 22)],

        u'水瓶座': [(1, 20), (2, 18)],

        u'双鱼座': [(2, 19), (3, 20)]

    }

    for k, v in xingzuo_map.items():

        if v[0] <= (month, day) <= v[1]:
            return k

    if (month, day) >= (12, 22) or (month, day) <= (1, 19):
        return u'摩羯座'

def super(xingzuo):
    xingzuosuper_map = {
        u'白羊座': "绿巨人\n白羊座的人本身就是，比较容易冲动的，是个直肠子的孩子，所以若是穿越科幻剧了，就会拥有绿巨人一样的超能力，一发怒就变身的那种，而且能力随着怒气值的增加而更强大。",

        u'金牛座': "变大变小\n金牛座可爱能吃，胃口很好的他们，经常能把小肚子吃得鼓鼓的，所以若是穿越科幻剧了，就会拥有蚁人那样的，变大变小的能力，可不要小瞧这个超能力哦！",

        u'双子座': "星爵\n双子座的人，嘴巴很是能说会道的，聊天能把你逗笑，吵架能把你气哭的那种，所以若是穿越科幻剧了，就会拥有像星爵一样的超能力，以神明之躯比肩凡人，“嘴炮”无敌，不过很可爱！",

        u'巨蟹座': "百发百中\n巨蟹座的人，大多眼神都是很好使的，视力很高，而且对色彩捕捉能力也很强，所以若是穿越科幻剧了，就会拥有像鹰眼一样的超能力，可谓是百发百中的！",

        u'狮子座': "吹口哨\n狮子座最是刀子嘴豆腐心了，明明就是个特别善良的人，但为了要面子，硬是把自己武装成狠人，所以若是穿越科幻剧了，就会成为勇度那样的超能力，吹吹口哨，就能放倒敌人。",

        u'处女座': "近身战\n处女座的人是很追求完美的，对自己要求也特别严苛，做任何事情，都必须做到最完美为止，所以若是穿越科幻剧了，就会拥有黑豹那样的超能力，不仅聪明智慧，而且近身战很强，动作行云流水~",

        u'天秤座': "无限复活\n天秤座的人，若是穿越科幻剧了，就会拥有小贱贱，那样的超能力，哪怕还剩一丢丢细胞，都可以无限复活的那种，因为天秤是最能绝地逢生的人了，他们不会被压垮，是无论如何都相信希望的人。",

        u'天蝎座': "魔法\n天蝎座的人要是对一件事情很感兴趣的话，那真的会彻夜不睡，去研究那个事情的，所以若是穿越科幻剧了，就会拥有像奇异博士那样的超能力，沉醉于学习魔法，最终也会成为强大的魔法师。",

        u'射手座': "高超智慧\n射手座的人是非常聪明的哦，逻辑思维能力和发散性思维能力都是数一数二的那种，因此数学特别棒的他们，若是穿越科幻剧了，就会凭借自己的高超智慧，成为像钢铁侠那样的人。",

        u'摩羯座': "雷神\n摩羯座的人内心信念是非常强大的，认定一件事情了，就会改变的那种，所以若是穿越科幻剧了，就会拥有像雷神那样的超能力，能够召唤雷电，而且正义柔情，是位很好的超能力者。",

        u'水瓶座': "镭射眼\n水瓶座的人，一般都拥有着一双犀利的眼睛，所以若是穿越科幻剧了，就会拥有像镭射眼那样的超能力，眼睛看到哪，就可以破坏哪里。",

        u'双鱼座': "心灵感应\n双鱼座的人第六感是非常强烈的，而且与人交往很是细心，天生就会洞察别人的心思，所以若是穿越科幻剧了，就会拥有心灵感应的能力，强者还能掌控别人的思绪呢！"
    }
    return xingzuosuper_map[xingzuo]


def img(xingzuo):
    xingzuofig_map = {
        u'白羊座': "1",

        u'金牛座': "2",

        u'双子座': "3",

        u'巨蟹座': "4",

        u'狮子座': "5",

        u'处女座': "6",

        u'天秤座': "7",

        u'天蝎座': "8",

        u'射手座': "9",

        u'摩羯座': "10",

        u'水瓶座': "11",

        u'双鱼座': "12"

    }
    # 图片保存的路径
    img_path = './imgs/' + xingzuofig_map[xingzuo] +'.png'

    return img_path


def xzyunshi(xingzuo):
    xingzuoen_map = {
        u'白羊座': "aries",

        u'金牛座': "taurus",

        u'双子座': "gemini",

        u'巨蟹座': "cancer",

        u'狮子座': "leo",

        u'处女座': "virgo",

        u'天秤座': "libra",

        u'天蝎座': "scorpio",

        u'射手座': "sagittarius",

        u'摩羯座': "capricorn",

        u'水瓶座': "aquarius",

        u'双鱼座': "pisces"

    }

    url = "http://api.tianapi.com/txapi/star/index"

    # 定义请求数据，并且对数据进行赋值
    values = {}
    values['key'] = 'XXX' ## 你自己申请的APIKEY
    values['astro'] = xingzuoen_map[xingzuo]

    # 对请求数据进行编码
    data = urllib.parse.urlencode(values).encode('utf-8')
    print(type(data))  # 打印<class 'bytes'>
    print(data)  # 打印b'status=hq&token=C6AD7DAA24BAA29AE14465DDC0E48ED9'

    # 若为post请求以下方式会报错TypeError: POST data should be bytes, an iterable of bytes, or a file object. It cannot be of type str.
    # Post的数据必须是bytes或者iterable of bytes,不能是str,如果是str需要进行encode()编码
    data = urllib.parse.urlencode(values)
    print(type(data))  # 打印<class 'str'>
    print(data)  # 打印status=hq&token=C6AD7DAA24BAA29AE14465DDC0E48ED9

    # 将数据与url进行拼接
    req = url + '?' + data
    # 打开请求，获取对象
    response = urllib.request.urlopen(req)
    print(type(response))  # 打印<class 'http.client.HTTPResponse'>
    # 打印Http状态码
    print(response.status)
    if response.status == 200:
        the_page = response.read()
        rsts = eval(the_page.decode("utf-8"))
        #print(rsts["newslist"])
        yunshi = []
        yunshi.append('综合指数：' + rsts["newslist"][0]["content"])
        yunshi.append('爱情指数：' + rsts["newslist"][1]["content"])
        yunshi.append('工作指数：' + rsts["newslist"][2]["content"])
        yunshi.append('财运指数：' + rsts["newslist"][3]["content"])
        yunshi.append('健康指数：' + rsts["newslist"][4]["content"])
        yunshi.append('幸运颜色：' + rsts["newslist"][5]["content"])
        yunshi.append('幸运数字：' + rsts["newslist"][6]["content"])
        yunshi.append('贵人星座：' + rsts["newslist"][7]["content"])
        yunshi.append('今日概述：' + rsts["newslist"][8]["content"])
        finalstr = ""
        for i in yunshi:
            finalstr += i+'\n'
        return finalstr


def match_input(input):
    print(input)
    with open("./predict.txt", "w", encoding="utf-8") as f:
        f.write(input + " 查星座\n")
    rst = match.start()
    return int(rst)

userstate = '0'


async def on_message(msg: Message):
    global userstate

    print(msg.talker().name)
    if msg.talker().name == '271828':
        # print(msg.talker().name)
        print("11",userstate)
        if userstate == '1-1':
            str = msg.text()
            print(str)
            rst = xingzuo(int(str[0]), int(str[2]))
            await msg.talker().say('你是' + rst)
            selfsuper = super(rst)
            await msg.talker().say('你星座的超能力是' + selfsuper)
            imgpath = img(rst)
            file_box_xz = FileBox.from_file(imgpath)
            await msg.talker().say(file_box_xz)
            yunshi = xzyunshi(rst)
            print(yunshi)
            userstate = '0'
            await msg.say("你的今日运势：\n" + yunshi)


        rst = match_input(msg.text())
        if rst == 1:
            userstate = '1-1'
            await msg.talker().say('请说出你的生日，格式如5.7、5月7日')
            await msg.say('不需要加年份哦')

        else:

            if msg.text() == 'ding':
                await msg.say('这是自动回复: dong dong dong')

            if msg.text() == 'hi' or msg.text() == '你好':
                await msg.say(
                    '这是自动回复: 现在很多年轻人都爱看科幻剧的，像是复仇者啊，里面有很多的英雄、超能力者，这些人都是我们的青春与情怀，那么12星座若穿越科幻剧了，会分别拥有什么超能力呢？\n机器人目前的功能是\n- 收到"属相", 根据提示回复你的属相\n- 收到"星座", 根据提示回复你的星座和今日运势还有在科幻世界你的超能力哦')

            if msg.text() == '属相':
                userstate = '2-1'
                await msg.say('请输入你的出生年份，请保持纯数字，如1998')
                
            if userstate == '2-1':
                year = msg.text()
                print(year)
                rst = chinese_shuxiang(int(year))
                await msg.say('你属' + rst)

                userstate = '0'

            else:
                rst = chat.chat(msg.text())
                await msg.say(rst)


async def on_scan(
        qrcode: str,
        status: ScanStatus,
        _data,
):
    print('Status: ' + str(status))
    print('View QR Code Online: https://wechaty.js.org/qrcode/' + qrcode)


async def on_login(user: Contact):
    print(user)


async def main():
    # 确保我们在环境变量中设置了WECHATY_PUPPET_SERVICE_TOKEN
    if 'WECHATY_PUPPET_SERVICE_TOKEN' not in os.environ:
        print('''
            Error: WECHATY_PUPPET_SERVICE_TOKEN is not found in the environment variables
            You need a TOKEN to run the Python Wechaty. Please goto our README for details
            https://github.com/wechaty/python-wechaty-getting-started/#wechaty_puppet_service_token
        ''')

    bot = Wechaty()

    bot.on('scan', on_scan)
    bot.on('login', on_login)
    bot.on('message', on_message)

    await bot.start()

    print('[Python Wechaty] Ding Dong Bot started.')


asyncio.run(main())
