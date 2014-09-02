WordVec
=======

WordVec是根据Google发布的word2vec个人理解后的一个c++重构版本。采用OpenMP的方式进行多线程训练。

##前置准备

* g++ 4.5.1 以上
* cmake 2.6 以上
* gflags 2.1.1 以上

新建build目录并对项目进行编译
		
	mkdir build
	cd build
	cmake ..
	make

之后在scripts目录下运行example.sh进行样例测试
	
	./example.sh


##训练输入
需要分词后以空白字符分割的大数据文本。
在data目录下有一个以text8_line_前缀的很小的example data，可以用来测试是否能跑通。


##代码说明
项目只有4个文件

* 其中distance.cpp与word2vec的distance.c一样
* wordvec.cpp 是算法核心，根据算法的理解重构了CBOW和Skip-Gram两种模型
* vocabulary 是对词库的载入，哈夫曼树的建立，低频次过滤等操作的封装
* utils.cpp 封装的是一些读取文件，遍历文件的操作


##参数说明
	-train			输入是训练文本所在路径
	-output			输出的词向量的二进制文本
	-hidden_size	神经网络隐含结点的数量，默认100
	-window			滑动窗口的大小，默认为5，这个窗口的是单边窗口尺寸。如果单边为5意味着大窗口尺寸是10
	-threads		多线程的数量，默认是4
	-cbow			选用CBOW(continuous bag of words)模型，与-skipgram不能同时开启
	-skipgram		选用skip-gram模型，与-cbow不能同时选用
	-sentence_size	缓存到内存的单词最大数量，默认1000
	-prefix			训练文本的前缀，因为多线程是按小文件并行，所以小文件可以定制一些前缀规则进行过滤
	
	
##脚本说明
	DATA_DIR=../data						#训练数据的文件夹目录
	BIN_DIR=../bin							#二进制文件目录
	VECTOR_DATA=$DATA_DIR/word-vector.bin	#词向量二进制模型的输出
	
	$BIN_DIR/WordVec  	-train $DATA_DIR 
						-output $VECTOR_DATA 
					   	-prefix text8_
						-hidden_size 200 
						-window 5 
						-threads 4 
						
	#以上脚本对DATA_DIR目录下的所有符合prefix规则的文件训练数据，并行训练。
	#训练的输出是word-vector.bin这一模型文件,训练结束后会在控制台输出训练的速度 word/thread/sec
	#通过运行 $BIN_DIR/distance $DATA_DIR/$VECTOR_DATA
	#调用distance这一可执行文件，用于计算词和词之间的相似性。

	
##注意事项
* 当前版本实现去掉了负采样(Negative Sampling)的部分,因为作者默认就没有开启，后人在实验过程中发现负采样并没有对效果有明显提升，开启负采样会增大训练时间。
* 当前版本实现去掉了原作者种存在的随机因素，如滑动窗口的过程中随机收缩窗口的大小，去掉后实验表明不影响效果。
* 建议使用cbow模型进行训练，速度比skip-gram快很多，对低频词的发现逊于skip-gram。
* 去掉了指数表的预处理，使得代码在算法逻辑上更加清晰易懂。




