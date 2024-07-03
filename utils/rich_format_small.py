emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}


def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + " " + s
	s = s + " " + emo_dict[emo]
	return s


if __name__ == "__main__":
    text = " <|zh|> This is a test"
    # text = "<|yue|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>你而家打个电话暂时<|yue|><|SAD|><|Speech|><|SPECIAL_TOKEN_13|>自一之后留低口述 marary sorry我拣咗做好人噶我就去见陈永人无论点都好我俾一个身份佢<|yue|><|SPECIAL_TOKEN_5|><|Speech|><|SPECIAL_TOKEN_13|>个档案喺我电脑里边密码系你生日日期<|nospeech|><|SPECIAL_TOKEN_5|><|SPECIAL_TOKEN_15|><|SPECIAL_TOKEN_13|><|nospeech|><|SPECIAL_TOKEN_5|><|SPECIAL_TOKEN_15|><|SPECIAL_TOKEN_13|><|nospeech|><|SPECIAL_TOKEN_5|><|SPECIAL_TOKEN_15|><|SPECIAL_TOKEN_13|><|nospeech|><|SPECIAL_TOKEN_5|><|SPECIAL_TOKEN_15|><|SPECIAL_TOKEN_13|><|yue|><|SAD|><|Speech|><|SPECIAL_TOKEN_13|>啲束手我都入过学校啊你卧底真系得意都系<|yue|><|SPECIAL_TOKEN_5|><|Speech|><|SPECIAL_TOKEN_13|>天我唔知得嚟我见得过我要嘅嘢我要嘅嘢你都未必带嚟啦<|yue|><|SPECIAL_TOKEN_5|><|Speech|><|SPECIAL_TOKEN_13|>咁即系点啊所嚟晒太阳噶嘛俾个机会我点俾机会你啊<|yue|><|SPECIAL_TOKEN_5|><|Speech|><|SPECIAL_TOKEN_13|>我以前冇得拣我而家想拣翻做好人好啊同法官讲啦<|yue|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>俾你做好人即系要死啊对唔住怪人啊<|ko|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>왜요 자연<|yue|><|ANGRY|><|BGM|><|SPECIAL_TOKEN_13|>放到两台先讲你一睇下何心卧底先佢喺我手度有咩事翻餐馆先讲放低上即刻放低上我报咗警啊我点解要信你啊你唔使信我<|nospeech|><|SPECIAL_TOKEN_5|><|SPECIAL_TOKEN_15|><|SPECIAL_TOKEN_13|>"
    # text = "<|ja|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>高校生探偵工藤信一幼馴人で同級生の毛ー利蘭ンと遊園地に遊びに行って黒づくめの男の怪しげな取引現場を目撃した<|ja|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>取引を見るのに夢中になっていた俺は背後から近づいてからもう一人の仲間に気づかなかった俺はその男に毒薬を飲まされ目が覚めたら体が縮んでしまっていた<|ja|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>工藤新一が生きていると奴らにバレたらまた命が狙われ周りの人間にも危害が及びアサ博士の助言で正体を隠すことに<|ja|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>俺は蘭に名前を聞かれて咄っ嗟に江戸川コナンと名乗り奴らの情報を掴かむために父親が探偵をやっている蘭ンの家に転がり込んだ<|ja|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>俺の正体を知っているのはア笠瀬博士俺の両親西野高校生探偵の服部平士同級生の灰原ラ愛ア笠瀬博士が小さくなった俺<|ja|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>のためにいろんな発明品を作ってくれたハ原は黒づくめの組織のメンバーだったが組織から逃げ出際俺が飲まされたのと同じ薬よ<|ja|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>んで体が縮んでしまったさらにもう一人解答キッとやが絡んでくると<|ja|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>面倒なことになるんだよ小さくなっても頭脳ンは同じ永久らしの目探偵真実は"
    text = "<|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>什么法人 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>什么看吧我的世界我来孵活 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>都说华流才是顶流而随着华语乐坛的崛起的确有不少华语歌手真正做到了用作品和歌声征服国际舞台那么本期视频就为小伙伴们盘点了这样火遍全球的四首华语歌曲话不多说快来看看有没有你喜欢的吧 <|nospeech|><|SPECIAL_TOKEN_5|><|SPECIAL_TOKEN_15|><|SPECIAL_TOKEN_13|> <|zh|><|NEUTRAL|><|Speech|><|SPECIAL_TOKEN_13|>number four play 我呸由蔡依林演唱发现于二零一四年是一首中西合并风格十分前卫的歌曲在这首歌中蔡依林可谓突破了自己以往的尺度特别是现场表演更是气场全开完全就是女王的风范 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>假求大中我呸快你是想情是风我呸快你是哪你的亚虫我呸我呸早配狗配 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>什么都什么都喜欢 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>number three 左手指月左手指月指指人心这是一首暗含佛家禅艺的歌曲除了精妙的作词之外歌曲超三个八度的高音也只有原唱萨顶鼎能演绎出其中的精髓而他的现场演唱更是让老外都惊羡不已 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>自然是你全带上回间 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>生 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>啊好爱我吗 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>number two 光年之外这是好莱坞大片太空旅客专程邀请邓紫棋为电影创作的主题曲而邓紫棋显然也不负他们所望这首光年之外不仅与电影的主题十分契合而且火爆全网成为了二零一七年的年度十大金曲果然华语小天后的魅力你真的可以永远相信 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>遥远在空之外 <|ja|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>伤後了没有你慢のち我疯狂跳 <|zh|><|SPECIAL_TOKEN_5|><|BGM|><|SPECIAL_TOKEN_13|>娘 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>number one 浮夸或许很多小伙伴不知道的是原创作者写这首歌其实一开始就是为了纪念哥哥张国荣后来被陈奕迅演唱后更是成为了一个经典浮夸式的演绎据说在二零一四年的某颁奖盛典因为 ethan 的现场太过浮夸以至于主办方不得不将这一段给剪掉 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>歇斯底里吧以眼泪流花吧一心只想你惊讶我旧是未存在不么从 <|zh|><|HAPPY|><|BGM|><|SPECIAL_TOKEN_13|>好了这就是本期节目的全部内容了喜欢的小伙伴别忘了点赞关注我们下期见拜拜"
    print("+"*10)
    print(format_str(text))
    print("+"*10)
    print(format_str_v2(text))
