好的 我现在要引入 @seps_modules_reviewed_v2_enhanced.py  里的模块进入这个项目。但是我要做如下修改：
  1.我只用SDTPS部分 其他的模块不需要
2. 具体来说 原模块是用图像自身 ，稀疏文本，稠密文本 生成四个score。然后 组合score 然后用这个 gumbel 生成 一些 weight matrix 用来选择token，
我们的改动是这样：1.我们的输入是 rgb nir tir 三种图像 所以 比如对于 rgb 来说 我们 用 sapn生成 rgb 自身的 score + nir全局特征对rgb的score+ tir全局特征对rgb的score + 图像自注意力的 Image-Salient (s^im) score
这四部分作为新的score 输入到 gumbel 生成 weight matrix 用来选择token， 对于 nir 和 tir 也是类似的操作。 也就是说我们改变了输入的源。其他组合方式不变。
   3. 具体的改动部分 ，在make_model.py里面 
                   RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                   NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                   TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))
   把这个就作为 要和对应patch做cross attn的特征，其他输入应该你们能看懂吧。如果你有不确定的地方可以问我，让我们一步步完成


1 是   Predictive Score (s^p) - MLP预测

  # 输入：每个patch自己的特征
  v_i  # (B, N, C) - patch特征

  # MLP网络
  s^p = Sigmoid(Linear(GELU(Linear(v_i, C//4)), 1))
      # (B,N,C) → (B,N,128) → (B,N,1) → (B,N)

  # 含义：MLP学习预测"这个patch本身有多重要"
  # 不依赖任何外部信息，纯粹看patch自己
你应该能在 seps_modules_reviewed_v2.py 里找到对应的代码实现

2.暂时就用这个形式

3.总之原则就是 我要用这个模块实现类似 token selection 和 增强的作用。 尽量保留原有的设计思路和过程。 这个机制要在hdm之前完成。或者说事实上 我们要用和这个机制替换掉 hdm 和atm也就是这俩都不要了，你可以写一个新的类或者有新的变量 让我们开启的时候就走这条路。



我有几个问题
1._compute_self_attention _compute_cross_attention 的计算 和原来的seps_modules_reviewed_v2.py 里的一样吗？ 你严格按照原来的写法吗？ 还是说有改动？ 你能帮我确认一下吗？里面是没有可学习参数吗？我不太确定 因为正常的 self attn 和 cross attn 是有可学习参数的。
2.         if self.use_gumbel:这里似乎不管用不用对最后的结果都没造成任何影响，因为我们已经得到了要选择出来的token了 如果想要这个发挥作用，应该怎么修改呢，你可以读一下tex文件这个文件是一个论文里面讲了我们借鉴的这个模块也就是sdtps的原理和设计思路，你帮我看看 这个use_gumbel到底是干嘛的，怎么发挥的作用
