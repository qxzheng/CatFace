from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp
"""
config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "vit_s_dp005_mask_0"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 384
config.optimizer = "adamw"
config.lr = 0.001
config.verbose = 2000
config.dali = False

#config.rec = "/home/zqx/data/faces_emore"
config.rec = "/home/zqx/data/faces_emore"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 15
config.warmup_epoch = config.num_epoch // 10
config.val_targets = []
"""

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "vit_b"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = True
config.weight_decay = 0.1
#config.batch_size = 256
config.batch_size = 512
config.optimizer = "adamw"
#config.optimizer = "sgd"
#config.lr = 0.001
config.lr = 0.001
config.verbose = 2000
config.dali = False

#config.rec = "/home/zqx/data/clear_before_20241031_align_112_0/cattrain"   #11W data 
#config.num_classes = 587
#config.num_classes = 610
#config.num_image = 5779
#config.num_image = 7112
'''
config.rec = "/home/zqx/pet-large-model/testCode/arcface/catface"   #11W data 
config.num_classes = 6363
config.num_image = 111247

config.rec = "/home/zqx/pet-large-model/testCode/arcface/Transface/catface"  #5W data
config.num_classes = 3239
config.num_image = 56631
'''
#config.num_classes = 6363
#config.num_image = 111247
config.num_epoch = 40
#config.num_epoch = 15
config.warmup_epoch = config.num_epoch // 10
config.val_targets = []
