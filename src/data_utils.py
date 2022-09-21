import random, os
import time
import numpy as np
import pickle as pkl

class Utils(object):
    def __init__(self):
        pass

    def load_raw_data_rating(self, path):
        f = open(path, 'r')
        user_historicals = {}
        user_dict = {}
        item_dict = {}
        interactive_length = {}
        user_count = 0
        item_count = 1

        raw_item_dict = {}

        categories_items = {}

        for line in f:
            data = line.split('::')
            user = int(data[0])
            item = int(data[1])
            rating = float(data[2])




            time_stmp = int(data[3][:-1])

            if user not in user_dict.keys():
                user_dict[user] = user_count
                user_count += 1

            if item not in item_dict.keys():
                item_dict[item] = item_count
                item_count += 1

            user = user_dict[user]

            raw_item_dict[item] = item_dict[item]
            item = item_dict[item]



            if user not in user_historicals.keys():
                user_historicals[user] = []
            user_historicals[user].append((user, item, rating, time_stmp))

        f.close()




        user_rating_matrix = {}

        for user in user_historicals.keys():
            user_historicals[user] = sorted(user_historicals[user], key=lambda a: a[-1])


            interactive_length[user] = len(user_historicals[user])


            user_rating_vector = np.zeros(shape=[item_count], dtype=np.uint8)

            for i in user_historicals[user]:
                user_rating_vector[i[1]] = i[2]

            user_historicals[user] = [d[1] for d in user_historicals[user]]
            user_rating_matrix[user] = user_rating_vector.copy()

        return user_historicals, user_rating_matrix, interactive_length

        return user_historicals, user_rating_matrix, interactive_length








