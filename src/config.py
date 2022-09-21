class Config():
    def __init__(self):

        self.EMBEDDING_SIZE = 50
        self.RNN_SIZE = 50
        self.HIDDEN_LAYER_SIZE = 50
        
        self.HISTORY_SIZE = 10

        self.LR = 25e-5
        self.REPLAY_BUFFER = 100000
        self.START_TRAINING_LIMIT = 1500
        
        self.UPDATE_LIMIT = 50
        self.user_batch_size = 20

        self.switch_step = 10

        self.BATCH_SIZE = 256
        self.DISCOUNT_FACTOR = 0.9
        
        
        self.DATA_SET = '1m'
        self.boundary_rating = 3.0

        self.CLUSTER_NUMS = 20

        self.ITEM_SIZE = 3707   

        self.MAX_STEPS = 20



