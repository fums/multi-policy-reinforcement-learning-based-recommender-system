import numpy as np

class Memory():
    def __init__(self, config):
        self.config = config
        self._current = 0
        self._count = 0

        self.p_state = np.empty(shape=[self.config.REPLAY_BUFFER, self.config.HISTORY_SIZE], dtype=np.int32)
        self.n_state = np.empty(shape=[self.config.REPLAY_BUFFER, self.config.HISTORY_SIZE], dtype=np.int32)

        self.p_state_next = np.empty(shape=[self.config.REPLAY_BUFFER, self.config.HISTORY_SIZE], dtype=np.int32)
        self.n_state_next = np.empty(shape=[self.config.REPLAY_BUFFER, self.config.HISTORY_SIZE], dtype=np.int32)

        self.action = np.empty(shape=[self.config.REPLAY_BUFFER], dtype=np.int32)


        self.reward = np.empty(shape=[self.config.REPLAY_BUFFER], dtype=np.float32)

        self.ban_items = np.empty(shape=[self.config.REPLAY_BUFFER, self.config.ITEM_SIZE], dtype=np.float32)
        self.ban_next_items = np.empty(shape=[self.config.REPLAY_BUFFER, self.config.ITEM_SIZE], dtype=np.float32)

        self.is_terminal = np.empty(shape=[self.config.REPLAY_BUFFER], dtype=np.bool)

        self.user = np.empty(shape=[self.config.REPLAY_BUFFER], dtype=np.int32)
        self.preuser = np.empty(shape=[self.config.REPLAY_BUFFER], dtype=np.int32)
        self.step = np.empty(shape=[self.config.REPLAY_BUFFER], dtype=np.float32)

        self.u_distribution = np.empty(shape=[self.config.REPLAY_BUFFER, self.config.CLUSTER_NUMS], dtype=np.float32)


        self.batch_size = self.config.BATCH_SIZE

    def add_record(self, p_state, n_state, p_state_next, n_state_next, action, ban_items, ban_next_items, reward, is_terminal, user,preuser,step, u_distribution):
        self.p_state[self._current] = p_state
        self.n_state[self._current] = n_state
        self.p_state_next[self._current] = p_state_next
        self.n_state_next[self._current] = n_state_next
        
        self.action[self._current] = action
        
        self.user[self._current] = user
        self.preuser[self._current] = preuser
        self.preuser[self._current] = preuser
        self.step[self._current] = step

        self.ban_items[self._current] = ban_items
        self.ban_next_items[self._current] = ban_next_items
        self.reward[self._current] = reward
        self.is_terminal[self._current] = is_terminal

        self.u_distribution[self._current] = u_distribution

        self._count = max(self._count, self._current + 1)
        self._current = (self._current + 1) % self.config.REPLAY_BUFFER



    def get_record(self, batch_size):
        index = np.random.choice(self._count, batch_size)
        batch_p_state = self.p_state[index]
        batch_n_state = self.n_state[index]
        batch_p_state_next = self.p_state_next[index]
        batch_n_state_next = self.n_state_next[index]
        batch_action = self.action[index]
        
        batch_user = self.user[index]
        batch_preuser = self.preuser[index]
        batch_step = self.step[index]

        bantch_ban_items = self.ban_items[index]
        bantch_ban_next_items = self.ban_next_items[index]
        batch_reward = self.reward[index]
        batch_is_terminal = self.is_terminal[index]
        batch_u_distribution = self.u_distribution[index]

        return batch_p_state, batch_n_state, batch_p_state_next, batch_n_state_next, batch_action, bantch_ban_items, bantch_ban_next_items, batch_reward, batch_is_terminal, batch_user, batch_preuser,batch_step, batch_u_distribution

    def get_user_batch(self, step_length):
        index = np.arange(0, step_length)
        batch_p_state = self.p_state[index]
        batch_n_state = self.n_state[index]
        batch_p_state_next = self.p_state_next[index]
        batch_n_state_next = self.n_state_next[index]
        batch_action = self.action[index]
        bantch_ban_items = self.ban_items[index]
        bantch_ban_next_items = self.ban_next_items[index]
        batch_reward = self.reward[index]
        batch_is_terminal = self.is_terminal[index]
        batch_user = self.user[index]
        batch_preuser = self.preuser[index]
        batch_step = self.step[index]
        batch_u_distribution = self.u_distribution[index]

        return batch_p_state, batch_n_state, batch_p_state_next, batch_n_state_next, batch_action, bantch_ban_items, bantch_ban_next_items, batch_reward, batch_is_terminal, batch_user,batch_preuser,batch_step, batch_u_distribution

    def get_all_record(self):
        index = np.arange(0, self._count)
        batch_p_state = self.p_state[index]
        batch_n_state = self.n_state[index]
        batch_p_state_next = self.p_state_next[index]
        batch_n_state_next = self.n_state_next[index]
        batch_action = self.action[index]
        bantch_ban_items = self.ban_items[index]
        bantch_ban_next_items = self.ban_next_items[index]
        batch_reward = self.reward[index]
        batch_is_terminal = self.is_terminal[index]
        batch_user = self.user[index]
        batch_preuser = self.preuser[index]
        batch_step = self.step[index]
        batch_u_distribution = self.u_distribution[index]

        return batch_p_state, batch_n_state, batch_p_state_next, batch_n_state_next, batch_action, bantch_ban_items, bantch_ban_next_items, batch_reward, batch_is_terminal, batch_user, batch_preuser,batch_step, batch_u_distribution

    def reset(self):
        self._current = 0
        self._count = 0

    def get_record_iter(self):
        while True:
            index = np.random.choice(self._count, self.batch_size)
            batch_p_state = self.p_state[index]
            batch_n_state = self.n_state[index]
            batch_p_state_next = self.p_state_next[index]
            batch_n_state_next = self.n_state_next[index]
            batch_action = self.action[index]
            bantch_ban_items = self.ban_items[index]
            bantch_ban_next_items = self.ban_next_items[index]
            batch_reward = self.reward[index]
            batch_is_terminal = self.is_terminal[index]
            batch_user = self.user[index]
            batch_preuser = self.preuser[index]
            batch_step = self.step[index]
            batch_u_distribution = self.u_distribution[index]

            yield batch_p_state, batch_n_state, batch_p_state_next, batch_n_state_next, batch_action, \
            bantch_ban_items, bantch_ban_next_items, batch_reward, batch_is_terminal, batch_user, batch_preuser,batch_step, batch_u_distribution

    @property
    def count(self):
        return self._count

































