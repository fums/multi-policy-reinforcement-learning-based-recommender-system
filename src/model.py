import tensorflow as tf
import numpy as np
import os
import random
import time


class DQNU(object):
    def __init__(self, config, user_emb_val = None, scope='', sess = None):
        self.config = config


        self.init_w = tf.contrib.layers.xavier_initializer(uniform=True)
        self.init_b = tf.constant_initializer(0.1)
        self.global_step = tf.Variable(tf.constant(0), name="global_step", trainable=False)

        user_emb_init = tf.constant_initializer(user_emb_val)
        user_size = user_emb_val.shape[0] - 1
        self.user_size = user_size
        user_feature_size  = user_emb_val.shape[1]
        self.user_feature_size = user_feature_size


        self.scope = scope

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        gpu_config.allow_soft_placement = True
        gpu_config.intra_op_parallelism_threads = 16
        gpu_config.inter_op_parallelism_threads = 16

        if sess is None:
          self.sess = tf.Session(config=gpu_config)
        else:
          self.sess = sess

        
        self.__build_placeholder()





        with tf.variable_scope(self.scope +"/usemb"):

          self.user_embeddings = tf.get_variable(shape=[user_size+1, user_feature_size],
                                                     dtype=tf.float32, trainable=False, initializer=user_emb_init,
                                                     name="user_embedding")

          self.user_emb = tf.reshape(tf.nn.embedding_lookup(self.user_embeddings, self.userID), [-1, user_feature_size])



        self.eval_q_value, self.user_scores = self.__build_model(self.positive_state, self.negative_state, self.user_emb,  trainable=True,
                                                  scope=self.scope + "action/eval")

        self.target_q_value, _ = self.__build_model(self.positive_state, self.negative_state, self.user_emb, trainable=False,
                                                 scope=self.scope + "action/target")

        self.eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+ 'action/eval')
        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope + 'action/target')


        self.assign_ops = []
        for e_p, t_p in zip(self.eval_params, self.target_params):
            self.assign_ops.append(tf.assign(ref=t_p, value=e_p))


        with tf.variable_scope("loss"):
            
            self.action_q = tf.batch_gather(self.eval_q_value, self.action)


            self.critic_loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.action_q))


            self.critic_train_op = tf.train.AdamOptimizer(learning_rate=self.config.LR, epsilon=1e-8).minimize(loss=self.critic_loss,
                                                                                                        global_step=self.global_step)

            self.user_loss = tf.reduce_mean( tf.batch_gather(-tf.log(1e-6 + self.user_scores), self.userID) )

            
            self.user_train_op = tf.train.AdamOptimizer(learning_rate=self.config.LR , epsilon=1e-8).minimize(loss=self.user_loss,
                                                                                                        global_step=self.global_step)
            




    def init_model(self):
        self.sess.run(tf.global_variables_initializer())
        self.update_target_params()






    def __build_placeholder(self):
        self.positive_state = tf.placeholder(dtype=tf.int32, shape=[None,10],
                                             name="positive_state")
        self.negative_state = tf.placeholder(dtype=tf.int32, shape=[None, 10],
                                             name="nagetive_state")

        self.action = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="action_index")
        self.userID = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="user_id")
        

        self.target_q = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="target_q")


    def __build_state_net(self, positive_states_id, negative_states_id, user_features, items_embedding, trainable, scope):

      with tf.variable_scope(scope):
            positive_state = tf.nn.embedding_lookup(items_embedding, positive_states_id)
            negative_state = tf.nn.embedding_lookup(items_embedding, negative_states_id)




            positive_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=self.config.EMBEDDING_SIZE, activation=tf.nn.tanh,
                                                       kernel_initializer=self.init_w, bias_initializer=self.init_b,
                                                       trainable=trainable, name="apositive_rnn_cell")

            negative_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=self.config.EMBEDDING_SIZE, activation=tf.nn.tanh,
                                                       kernel_initializer=self.init_w, bias_initializer=self.init_b,
                                                       trainable=trainable, name="anegative_rnn_cell")



            with tf.variable_scope("positive_feature"):

                p_out, __ = tf.nn.dynamic_rnn(cell=positive_rnn_cell, 
                                              inputs=positive_state, dtype=tf.float32)



                p_out = tf.transpose(p_out, perm=[1, 0, 2])

                p_out = p_out[-1]

                p_features = tf.layers.dense(inputs=p_out, units=self.config.HIDDEN_LAYER_SIZE,
                                             activation=None, kernel_initializer=self.init_w,
                                             bias_initializer=self.init_b,
                                             trainable=trainable, name="p_features_1")


                p_features = tf.nn.relu(p_features)

            with tf.variable_scope("negative_feature"):

                n_out, __ = tf.nn.dynamic_rnn(cell=negative_rnn_cell, 
                                              inputs=negative_state, dtype=tf.float32)
                n_out = tf.transpose(n_out, perm=[1, 0, 2])
                n_out = n_out[-1]

                n_features = tf.layers.dense(inputs=n_out, units=self.config.HIDDEN_LAYER_SIZE,
                                             activation=None, kernel_initializer=self.init_w,
                                             bias_initializer=self.init_b, trainable=trainable,
                                             name="n_features_1")
              
                n_features = tf.nn.relu(n_features)



            with tf.variable_scope("mix_feature"):
                user_features = tf.layers.dense(inputs=user_features, units=self.config.HIDDEN_LAYER_SIZE,
                             activation=tf.nn.relu, kernel_initializer=self.init_w,
                             bias_initializer=self.init_b, trainable=trainable,
                             name="u_features")


      return p_features, n_features, user_features



    def __build_model(self, positive_states_id, negative_states_id, user_features, trainable, scope):

        with tf.variable_scope(scope):
          with tf.variable_scope('embedding'):


            items_embedding = tf.get_variable(shape=[self.config.ITEM_SIZE, self.config.EMBEDDING_SIZE],
                                                   dtype=tf.float32, trainable=trainable, initializer=self.init_w,
                                                   name="items_embedding")

          with tf.variable_scope('critic_net'):
            p_features, n_features, user_features = self.__build_state_net(positive_states_id, negative_states_id, user_features, items_embedding, trainable, 'critic_net')


            discrimitor_features = tf.concat([p_features, n_features], axis=-1)
            
            discrimitor_features = tf.layers.dense(inputs=discrimitor_features, units=self.config.HIDDEN_LAYER_SIZE,
                                           activation=None, kernel_initializer=self.init_w,
                                           bias_initializer=self.init_b, trainable=trainable,
                                           name="user_features_layer")

            user_scores = tf.layers.dense(inputs=discrimitor_features, units= self.user_size,
                               activation=tf.nn.softmax, kernel_initializer=self.init_w,
                               bias_initializer=self.init_b, trainable=trainable,
                               name="user_net")


            mix_features = tf.concat([p_features, n_features, user_features], axis=-1)

            mix_features = tf.layers.dense(inputs=mix_features, units=self.config.HIDDEN_LAYER_SIZE,
                                           activation=None, kernel_initializer=self.init_w,
                                           bias_initializer=self.init_b, trainable=trainable,
                                           name="mixed_features_layer")

            mix_features = tf.nn.relu(mix_features)


            q_value = tf.layers.dense(inputs=mix_features, units=self.config.ITEM_SIZE,
                               activation=None, kernel_initializer=self.init_w,
                               bias_initializer=self.init_b, trainable=trainable,
                               name="qnet")




        return q_value, user_scores



    def choose_batch_action(self, p_states_id, n_states_id, ban_items, users):
        actions = []

        q_value = self.sess.run(self.eval_q_value,
                                feed_dict={
                                    self.positive_state: np.reshape(p_states_id, [-1, self.config.HISTORY_SIZE]),
                                    self.negative_state: np.reshape(n_states_id, [-1, self.config.HISTORY_SIZE]),
                                    self.userID: users
                                    })
        assert np.isnan(q_value).sum() == 0 
        for _i in range(len(p_states_id)):
            q_value[_i][np.asarray(ban_items[_i]) == 0.0] = -np.inf
            action = np.argmax(q_value[_i])
            actions.append(action)

        return actions





    def get_next_all_action_q(self, p_state_next, n_state_next, ban_next_items, users):

        ban_next_items = np.asarray(ban_next_items)
        next_q = self.sess.run(self.target_q_value, feed_dict={self.positive_state: p_state_next,
                                                               self.negative_state: n_state_next,
                                                               self.userID: users
                                                               })
        next_q[ban_next_items == 0.0] = -np.inf
        next_q = np.max(next_q, axis=-1)

        return next_q






    def learn_discrimitor(self,  p_states_id, n_states_id, users):
        actions = []

        ul, _ = self.sess.run([self.user_loss, self.user_train_op],
                                feed_dict={
                                    self.positive_state: np.reshape(p_states_id, [-1, self.config.HISTORY_SIZE]),
                                    self.negative_state: np.reshape(n_states_id, [-1, self.config.HISTORY_SIZE]),
                                    self.userID: users
                                    })


        return ul




    def pred_user(self,  p_states_id, n_states_id):
        users = []

        user_scores = self.sess.run(self.user_scores,
                                feed_dict={
                                    self.positive_state: np.reshape(p_states_id, [-1, self.config.HISTORY_SIZE]),
                                    self.negative_state: np.reshape(n_states_id, [-1, self.config.HISTORY_SIZE])
                                    })

        assert np.isnan(user_scores).sum() == 0 

        for _i in range(len(p_states_id)):

            user = np.argmax(user_scores[_i])
            users.append(user)

        return users, user_scores




    def learn_critic(self,  p_state, n_state, p_state_next, n_state_next, action, ban_items, ban_next_items, reward, users):


        target_q = self.get_next_all_action_q(p_state_next, n_state_next, ban_next_items, users)
        assert np.isnan(target_q).sum() == 0 
        assert np.isnan(reward).sum() == 0 

        target = (reward + self.config.DISCOUNT_FACTOR * target_q).reshape([-1,1])
        assert np.isnan(target).sum() == 0 

        _, loss, step = self.sess.run([self.critic_train_op, self.critic_loss, self.global_step], feed_dict={self.positive_state: p_state,
                                                                self.negative_state: n_state,
                                                                self.action: action.reshape([-1,1]),
                                                                self.target_q: target,
                                                                self.userID: users
                                                               })
        
        return step, loss






    def learn_acnet(self, p_state, n_state, p_state_next, n_state_next, action, ban_items, ban_next_items, reward, users):
      step, loss = self.learn_critic(p_state, n_state, p_state_next, n_state_next, action, ban_items, ban_next_items, reward, users)

      if step % self.config.UPDATE_LIMIT == 0:
          self.update_target_params()
      return loss


    def update_target_params(self):
        self.sess.run(self.assign_ops)










class BPR_MF(object):

    def __init__(self, sess, user_count, item_count, hidden_dim, lr):
      gpu_config = tf.ConfigProto()
      gpu_config.gpu_options.allow_growth = True
      gpu_config.allow_soft_placement = True
      gpu_config.intra_op_parallelism_threads = 16
      gpu_config.inter_op_parallelism_threads = 16

      
      self.sess = sess

      if sess is None:
          self.sess = tf.Session(config=gpu_config)


      
      self.lr = lr
      self.u = tf.placeholder(tf.int32, [None])
      self.i = tf.placeholder(tf.int32, [None])
      self.j = tf.placeholder(tf.int32, [None])

      user_emb_w = tf.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                   initializer=tf.random_normal_initializer(0, 0.1))
      item_emb_w = tf.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                   initializer=tf.random_normal_initializer(0, 0.1))
      self.user_emb_w = user_emb_w
      u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)
      i_emb = tf.nn.embedding_lookup(item_emb_w, self.i)

      j_emb = tf.nn.embedding_lookup(item_emb_w, self.j)
      x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keepdims=True)
      l2_norm = tf.add_n([tf.reduce_sum(tf.multiply(u_emb, u_emb)),
                          tf.reduce_sum(tf.multiply(i_emb, i_emb)),
                          tf.reduce_sum(tf.multiply(j_emb, j_emb))])

      regulation_rate = 0.0001
      self.bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))
      self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.bprloss)
      u_emb2 = tf.reshape(u_emb, (-1, 1, hidden_dim))
      i_emb2 = tf.reshape(i_emb, (-1, 1, hidden_dim))
      self.pred = tf.matmul(u_emb2, i_emb2, transpose_b=True)

    def train(self, batch_users, batch_items, batch_neg_items):

      feed_dict = {self.u: batch_users,
                   self.i: batch_items,
                   self.j: batch_neg_items}

      _, current_loss = self.sess.run([self.train_op, self.bprloss], feed_dict=feed_dict)

      return current_loss

    def predict(self, data, batch_size=0, verbose=0):      
      users, items = data
      feed_dict = {self.u: users, self.i: items}
      pred = self.sess.run(self.pred, feed_dict=feed_dict)
      pred = pred.reshape([1, len(items)])

      return pred
    def get_userembeddings(self):
      W = self.sess.run(self.user_emb_w)
      return W
    def init(self):

      self.sess.run(tf.global_variables_initializer())




