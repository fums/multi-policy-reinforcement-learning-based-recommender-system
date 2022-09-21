from __future__ import print_function
import numpy as np
import os

from config import Config

from model import BPR_MF as BPR
from model import  DQNU
from data_utils import Utils
from memory import Memory
import random
import time
import tensorflow as tf
from sklearn.cluster import KMeans

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



fast_norm_ratings= {}
def get_norm_rating(r, r_small, r_large):
    if r in fast_norm_ratings:
        return fast_norm_ratings[r]
    a = 2.0 / (float(r_large) - float(r_small)) #0.4
    b = - (float(r_large) + float(r_small)) / (float(r_large) - float(r_small)) # -1.0

    reward = a * r + b
    fast_norm_ratings[r] = reward
    return reward




class User_Enviroment(object):
    def __init__(self, user_history, user_rating_matrix,  user_cluster_map,cluster_nb, cfg):
        self.user_history = user_history
        self.cfg = cfg
        self.user_rating_matrix = user_rating_matrix
        self.user_cluster_map = user_cluster_map
        self.cluster_nb = cluster_nb



        self.rating_matrix = self.get_init_env()

    def get_init_env(self):
        self.user_nb = len(self.user_history)
        self.item_nb = cfg.ITEM_SIZE
        self.rating_matrix = np.zeros([self.user_nb, self.item_nb]) + get_norm_rating(0.0, 0.0, 5.0)
        self.user_index = np.arange(0,self.user_nb)
        self.user_cluster_matrix = np.zeros([self.user_nb, self.cluster_nb])

        user_count = 0

        for u in self.user_history:
            items = self.user_history[u]
            for i in items:
                self.rating_matrix[user_count][i] = get_norm_rating(self.user_rating_matrix[u][i], 0.0, 5.0) 
            self.user_cluster_matrix[user_count][self.user_cluster_map[u]] = 1.0
            user_count += 1


        self.init_cluster_p =  np.sum(self.user_cluster_matrix, axis = 0) / self.user_nb

        self.init_cluster_entropy =  - np.sum(self.init_cluster_p * np.log(self.init_cluster_p + 1e-6))
        self.last_distribution = self.init_cluster_p 

        return self.rating_matrix





    def get_info(self, action, feedback):
        pos_users, pos_nb, neg_users, neg_nb, user_positive_cluster_entropy, user_negative_cluster_entropy, info_gain = self.get_pos_and_neg_users_w_feedback(action, feedback)
        #print ('!!!!!!')
        self.current_distribution = self.last_distribution
        if feedback > 0:
            self.user_index = pos_users
            self.last_state_entropy = user_positive_cluster_entropy
            self.last_distribution = self.user_positive_cluster_distribution
        else:
            self.user_index = neg_users
            self.last_state_entropy = user_negative_cluster_entropy
            self.last_distribution =  self.user_negative_cluster_distribution
        if info_gain > 0.2:
            info_gain = 0.2
        if info_gain <-0.2:
            info_gain = -0.2
        return info_gain







    def get_pos_and_neg_users_w_feedback(self, action, feedback):
        total_user_nb =  float(self.user_index.shape[0]) 
        pos_mat = np.where(self.env[:, action] > 0.0)
        
        pos_users = np.intersect1d(self.user_index,pos_mat)
        pos_nb = float(pos_users.shape[0])

        neg_users = np.setdiff1d(self.user_index,pos_mat)

        neg_nb =  float(neg_users.shape[0])
        assert (neg_nb + pos_nb == total_user_nb)


        if pos_nb >0:
            user_positive_cluster_distribution =  np.sum(self.user_cluster_matrix[pos_users, :], axis = 0) / pos_nb
        else:
            user_positive_cluster_distribution = np.array([1.0/self.cluster_nb] * self.cluster_nb)


        if neg_nb > 0:
            user_negative_cluster_distribution = np.sum(self.user_cluster_matrix[neg_users, :], axis = 0) / neg_nb
        else:
            user_negative_cluster_distribution = np.array([1.0/self.cluster_nb] * self.cluster_nb)


        user_positive_cluster_entropy =  - np.sum(user_positive_cluster_distribution * np.log(user_positive_cluster_distribution + 1e-6))
        user_negative_cluster_entropy =  - np.sum(user_negative_cluster_distribution * np.log(user_negative_cluster_distribution + 1e-6))


        if feedback > 0:
            info_gain = self.last_state_entropy - user_positive_cluster_entropy 
        else:
            info_gain = self.last_state_entropy - user_negative_cluster_entropy 



        self.user_positive_cluster_distribution = user_positive_cluster_distribution
        self.user_negative_cluster_distribution = user_negative_cluster_distribution
        
        return pos_users, pos_nb, neg_users, neg_nb, user_positive_cluster_entropy, user_negative_cluster_entropy, info_gain


    def reset_env(self):

        self.env = self.rating_matrix
        ban_actions = np.ones([self.item_nb,])
        new_pos_state = [0] * self.cfg.HISTORY_SIZE
        new_neg_state = [0] * self.cfg.HISTORY_SIZE
        self.user_index = np.arange(0,self.user_nb)
        self.last_state_entropy =  self.init_cluster_entropy
        self.last_distribution = self.init_cluster_p 

        return new_pos_state, new_neg_state, ban_actions 





def compute_ndcg(labels, true_labels):
    dcg_labels = np.array(labels)
    dcg = np.sum(dcg_labels / np.log2(np.arange(2, dcg_labels.size + 2)))

    idcg_labels = np.array(true_labels)
    idcg = np.sum(idcg_labels / np.log2(np.arange(2, idcg_labels.size + 2)))
    if not idcg:
        return 0

    return dcg / idcg


def compute_batch_ndcg(hit_seqs):

    batch_size, seq_size = hit_seqs.shape
    idcg_seq = np.ones([seq_size,])
    idcg = np.sum(idcg_seq / np.log2(np.arange(2, idcg_seq.size + 2)))
    ndcg_res = []
    for i in range(batch_size):
        dcg = np.sum(hit_seqs[i] / np.log2(np.arange(2, hit_seqs[i].size + 2)))
        ndcg = dcg/idcg
        ndcg_res.append(ndcg)
    return ndcg_res


def load_dataset(utils, cfg):

    user_historicals, user_rating_matrix, interactive_length = utils.load_raw_data_rating("../data/" + cfg.DATA_SET + "_raw")
    
    return user_historicals, interactive_length, user_rating_matrix

def generate_a_negative_sample(user, user_history, item_num):
    j = random.randint(1, item_num)
    while j in user_history[user]:
        j = random.randint(1, item_num)
    return j


def train_bpr(utils, cfg, user_rating_matrix, user_keys):
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    user_historicals, interactive_length, user_rating_matrix = load_dataset(utils, cfg)
    lr = 0.005
    users = [ u for u in user_keys]
    
    model = BPR(None, len(user_historicals), cfg.ITEM_SIZE, cfg.HIDDEN_LAYER_SIZE, lr)
    model.init()
    item_count = cfg.ITEM_SIZE
    for epoch in range(20):
        np.random.shuffle(users)
        batch_users = []
        batch_items = []
        batch_neg_items = []
        losses = []
        for user in users:
            interactions = [_i for _i in user_historicals[user] if user_rating_matrix[user][_i] >= cfg.boundary_rating]
            user_selected_items_size = len(interactions)
            for l, item in enumerate(interactions):
                negative_item = generate_a_negative_sample(user, user_historicals, item_count)
                batch_users.append(user)
                batch_items.append(item)
                batch_neg_items.append(negative_item)

                if len(batch_users) >= 256 or l == user_selected_items_size - 1:
                    current_batch_size = len(batch_users)

                    # train model
                    current_loss = model.train(batch_users, batch_items, batch_neg_items)
                    losses.append(current_loss)
                    batch_items = []
                    batch_neg_items = []
                    batch_users = []


    user_embs = model.get_userembeddings()

    return user_embs

def get_cluster_embeddings(embs,train_user_embs, k=10):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(train_user_embs)
    cluster_ids = kmeans.predict(embs)

    return cluster_ids, kmeans.cluster_centers_








def trainBatchPopUserHQlearning(utils, cfg):


    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    highest_hit = 0.0
    highest_epoch = 0

    user_historicals, interactive_length, user_rating_matrix = load_dataset(utils, cfg)

    
    train_user_id = []
    test_user_id = []
    val_user_id = []

    for user in user_historicals:
        r = random.random()
        if r <0.8:
            train_user_id.append(user)
        elif r>=0.8 and r <0.9:
            val_user_id.append(user)
        else:
            test_user_id.append(user)

    print ('!!!!!!! BPR start')
    user_embs = train_bpr(utils, cfg, user_rating_matrix, train_user_id)

    print ('!!!!!!! BPR done')

    train_user_embs = np.array([user_embs[i] for i in range(len(user_embs)) if i in train_user_id])
    
    user_cluster_nb = cfg.CLUSTER_NUMS
    user_cluster_map, cluster_embs =  get_cluster_embeddings(user_embs, train_user_embs, user_cluster_nb)



    center_user_emb = np.mean(train_user_embs, axis=0)
    cluster_embs = np.row_stack((cluster_embs, center_user_emb))



    train_user_history = {}
    test_user_history = {}
    user_batch_size = cfg.user_batch_size

    for user in train_user_id:
        train_user_history[user] = user_historicals[user]

    for user in test_user_id:
        test_user_history[user] = user_historicals[user]

    train_envs = []
    for _ in range(user_batch_size):
        train_env = User_Enviroment(train_user_history, user_rating_matrix,  user_cluster_map, user_cluster_nb, cfg)
        
        train_envs.append(train_env)

    
    amodel = DQNU(cfg, cluster_embs, scope='dq1')

    amodel.init_model()

    a_memory = Memory(cfg)

    a_mem_iter = a_memory.get_record_iter()

    step = 0
    
    MAX_STEPS = cfg.MAX_STEPS


    switch_step = cfg.switch_step
    
    total_training_episodes_nb = 500000
    

    evaluate_point = 5000

    episode_count = 0
    np.random.shuffle(train_user_id)
    t_train_user_id = train_user_id
    user_nb = len(t_train_user_id)
    batch_users_nb = user_batch_size
    lhs = 0
    rhs = lhs + user_batch_size

    while episode_count < total_training_episodes_nb:
        
        train_user_total_hit = []
        train_total_ndcg = []
        test_user_total_hit = []
        test_user_phase1_hit = []
        test_user_phase2_hit = []
        test_total_ndcg = []
        all_user_ndcg = []

        count = 0
        

        
        next_evaluate_point = episode_count + evaluate_point

        all_sum_reward = []
        train_loss = []
        while episode_count < next_evaluate_point:

            
            episode_count += user_batch_size
            batch_users = t_train_user_id[lhs:rhs]
            np_batch_users = np.reshape(np.array( [user_cluster_map[_u] for _u in batch_users] ), [-1,1])
            batch_users_nb = len(batch_users)
            sum_rewards = [0.0] * batch_users_nb
            p_state = []
            n_state = []
            ban_items = []

            for _i in range(batch_users_nb):
                _p_state, _n_state, _ = train_envs[_i].reset_env()
                p_state.append(_p_state)
                n_state.append(_n_state)

            ban_items = np.ones(shape=[batch_users_nb, cfg.ITEM_SIZE], dtype=np.float32)
            ban_next_items = ban_items.copy()
            terminal = False
            inner_step = 0

            for _step in range(MAX_STEPS):                
                rewards = []

                if _step <switch_step:
                    acutal_batch_pred_users = np.array([user_cluster_nb] * batch_users_nb, dtype='int32')
                if _step ==switch_step:
                    batch_pred_users, batch_pred_users_score = amodel.pred_user(p_state, n_state)
                    acutal_batch_pred_users = batch_pred_users  

                batch_pred_users = acutal_batch_pred_users

                batch_pred_users =  np.reshape(batch_pred_users, [-1,1])

                choosed_items = amodel.choose_batch_action(p_state, n_state, ban_items, batch_pred_users)

                p_state_next = [[_i for _i in p_state[_j]]  for _j in range(batch_users_nb)]
                n_state_next = [[_i for _i in n_state[_j]]  for _j in range(batch_users_nb)]

                for _j in range(batch_users_nb):


                    user_id = batch_users[_j]
                    choosed_item = choosed_items[_j]

                    rating = user_rating_matrix[user_id][choosed_item]
                    norm_rating = get_norm_rating(rating, 0.0, 5.0)

                    info_gain = train_envs[_j].get_info(choosed_item, norm_rating)
                    
                    if choosed_item in user_historicals[user_id] and rating>=cfg.boundary_rating:
                        p_state_next[_j].pop(0)
                        p_state_next[_j].append(choosed_item)
                    else:
                        n_state_next[_j].pop(0)
                        n_state_next[_j].append(choosed_item)

                    ban_next_items[_j][choosed_item] = 0.0
                    rewards.append(norm_rating + info_gain)

                for _j in range(batch_users_nb):
                    a_memory.add_record(p_state[_j], n_state[_j], p_state_next[_j], n_state_next[_j], choosed_items[_j], ban_items[_j], ban_next_items[_j], rewards[_j], terminal, np_batch_users[_j], batch_pred_users[_j], inner_step, train_envs[_j].current_distribution )
                
                ban_items =ban_next_items.copy() 
                p_state = [[_i for _i in p_state_next[_j]] for _j in range(batch_users_nb)]
                n_state = [[_i for _i in n_state_next[_j]] for _j in range(batch_users_nb)]


                step += 1
                inner_step += 1

            if a_memory.count > cfg.START_TRAINING_LIMIT:
                

                batch_p_state, batch_n_state, batch_p_state_next, batch_n_state_next, batch_action, \
                bantch_ban_items, bantch_ban_next_items, batch_reward, batch_is_terminal, learn_batch_users, learn_batch_pre_users, batch_steps, batch_u_distributions = next(a_mem_iter)

                learn_batch_users = np.reshape(learn_batch_users, [-1,1])
                learn_batch_pre_users = np.reshape(learn_batch_pre_users, [-1,1])
                

                loss = amodel.learn_acnet(batch_p_state, batch_n_state, batch_p_state_next, batch_n_state_next, batch_action, bantch_ban_items, bantch_ban_next_items, batch_reward, learn_batch_pre_users)

                train_loss.append(loss)

                ul = amodel.learn_discrimitor(batch_p_state, batch_n_state, learn_batch_users)

            lhs = rhs
            if lhs >= user_nb-1:
                lhs = 0

            rhs = lhs + user_batch_size

        # for test

        test_lhs = 0   
        test_rhs = user_batch_size
        test_rewards = []

        while test_rhs < len(test_user_id):
            batch_users = test_user_id[test_lhs: test_rhs]
            batch_users_nb = len(batch_users)
            hit = [0] * batch_users_nb
            sum_rewards = [0.0] * batch_users_nb

            phase1_hit = [0] * batch_users_nb
            phase2_hit = [0] * batch_users_nb

            hit_seq = np.zeros([batch_users_nb, MAX_STEPS])


            p_state = [[0 for _ in range(cfg.HISTORY_SIZE)] for _ in range(batch_users_nb)]
            n_state = [[0 for _ in range(cfg.HISTORY_SIZE)] for _ in range(batch_users_nb)]

            ban_items = np.ones(shape=[batch_users_nb, cfg.ITEM_SIZE], dtype=np.float32)

            terminal = False

            inner_step = 0
            hit_history = [[] for _ in range(batch_users_nb)]
            for _step in range(MAX_STEPS):

                batch_pred_users, pre_scores = amodel.pred_user(p_state, n_state)

                if _step <switch_step:

                    acutal_batch_pred_users = np.array([user_cluster_nb] * batch_users_nb, dtype='int32')

                elif _step == switch_step:
                    acutal_batch_pred_users = batch_pred_users 

                batch_pred_users = acutal_batch_pred_users

                batch_pred_users = np.reshape(batch_pred_users, [-1,1])
                
                batch_choosed_items = amodel.choose_batch_action(p_state, n_state, ban_items, batch_pred_users)


                
                for _j in range(batch_users_nb):

                    user_id = batch_users[_j]
                    choosed_items = batch_choosed_items[_j]


                    rating = user_rating_matrix[user_id][choosed_items]
                    norm_rating = get_norm_rating(rating, 0.0, 5.0)
                    sum_rewards[_j] += norm_rating

                    if choosed_items in user_historicals[user_id] and rating>=cfg.boundary_rating:
                        p_state[_j].pop(0)
                        p_state[_j].append(choosed_items)
                        ban_items[_j][choosed_items] = 0.0
                        hit[_j] += 1
                        hit_seq[_j][inner_step] = 1.0
                        hit_history[_j].append(choosed_items)
                    else:
                        n_state[_j].pop(0)
                        n_state[_j].append(choosed_items)                        
                        ban_items[_j][choosed_items] = 0.0

                    if _step < switch_step:
                        phase1_hit[_j] += hit_seq[_j][inner_step]
                    else:                        
                        phase2_hit[_j] += hit_seq[_j][inner_step]
                
                inner_step += 1

            ndcg_res = compute_batch_ndcg(hit_seq)

            for _j in range(len(hit)):
                test_user_total_hit.append(float(hit[_j]) / MAX_STEPS)
                
                test_user_phase1_hit.append(float(phase1_hit[_j]) / switch_step)
                test_user_phase2_hit.append(float(phase2_hit[_j]) / (MAX_STEPS-switch_step))

                test_total_ndcg.append(ndcg_res[_j])
                test_rewards.append(sum_rewards[_j])

                assert (len(hit_history[_j]) == len(set(hit_history[_j])))

 

            
            test_lhs = test_rhs
            test_rhs = test_lhs + user_batch_size


        if highest_hit < np.mean(test_user_total_hit):

            highest_hit = np.mean(test_user_total_hit)
            highest_epoch = episode_count


        print("Episode_count:", episode_count)
        print ('PID, ', os.getpid())
        print ('MAX_STEPS', cfg.MAX_STEPS)
        print ('item', cfg.ITEM_SIZE)
        print("dataset:", cfg.DATA_SET)
        print("highest_epoch:", highest_epoch, "highest_hit: ", highest_hit)
        print("avg test hit:", np.mean(test_user_total_hit))
        print ('avg train loss:', np.mean(train_loss))
        print("avg test phase1 hit:", np.mean(test_user_phase1_hit))
        print("avg test phase2 hit:", np.mean(test_user_phase2_hit))
        print ("avg test ndcg:", np.mean(test_total_ndcg))
        print ("avg test sum reward:", np.mean(test_rewards))
        
        print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )

        amodel.update_target_params()



    return amodel







if __name__ == "__main__":

    cfg = Config()
    utils = Utils()

    amodel =  trainBatchPopUserHQlearning(utils, cfg)





