import mindpet
# 后来使用脚本进行微调，没有用到这个程序，未来可以对这个程序进行完善
# 搭建链接 MindFormers 和 MindPet 之间的联系桥梁


from mindpet.delta import LoRADense

#定义LoRADense对象
dense1 = LoRADense(in_channels=1*28*28, 
                   out_channels=512, 
                   lora_rank=8, 
                   lora_alpha=16)

#通过分片以便分布式训练
dense1.shard(strategy_org_dense_matmul=((2, 1), (4, 1)), 
             strategy_org_bias_add=((2, 4), (4,)),
             strategy_lora_dropout=((2, 1),), 
             strategy_lora_a_matmul=((2, 1), (1, 1)), 
             strategy_lora_b_matmul=((2, 1), (4, 1)),
             strategy_lora_mul=((2, 4), ()), 
             strategy_lora_add=((2, 4), (2, 4)),
             strategy_activation=((2, 4), (2, 4)))

