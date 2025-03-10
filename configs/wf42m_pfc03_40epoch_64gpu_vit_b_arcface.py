from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "vit_b"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 384
#config.optimizer = "adamw"
config.optimizer = "sgd"
config.lr = 0.001
config.verbose = 2000
config.dali = False

config.rec = "/home/zqx/data/faces_emore"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 15
config.warmup_epoch = config.num_epoch // 10
config.val_targets = []
