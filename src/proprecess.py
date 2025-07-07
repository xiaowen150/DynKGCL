import argparse
import numpy as np
from scipy.sparse import  dok_matrix
from sklearn.metrics.pairwise import cosine_similarity
import glob


RATING_FILE_NAME = dict({'movie1M': 'ratings.dat', 'book': 'BX-Book-Ratings.csv', 'music': 'user_artists.dat'})
SEP = dict({'movie1M': '::', 'book': ';', 'music': '\t'})
THRESHOLD = dict({'movie1M': 4, 'book': 0, 'music': 0})


def read_item_index_to_entity_id_file():
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def convert_rating():
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = {}
    user_neg_ratings = {}
    max_pos = 10

    if DATASET in ['music', 'book']:
        max_pos = 10
    elif DATASET == 'movie1M':
        max_pos = 50
    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:
            continue
        item_index = item_index_old2new[item_index_old]
        user_index_old = int(array[0])
        rating = float(array[2])

        if rating >= THRESHOLD[DATASET]:
            user_pos_ratings.setdefault(user_index_old, set()).add(item_index)
        else:
            user_neg_ratings.setdefault(user_index_old, set()).add(item_index)

    user_index_old2new = {u: i for i, u in enumerate(user_pos_ratings)}
    user_cnt = len(user_index_old2new)

    n_user = user_cnt
    n_item = len(item_index_old2new)
    embedding_dim = args.dim

    user_embeddings = np.random.rand(n_user, embedding_dim)
    item_embeddings = np.random.rand(n_item, embedding_dim)

    batch_size = 1000
    for batch_start in range(0, len(user_pos_ratings), batch_size):
        batch_users = list(user_pos_ratings.keys())[batch_start:batch_start + batch_size]
        batch_pos_ratings = {user: user_pos_ratings[user] for user in batch_users}

        output_file = f'../data/{DATASET}/contrastive_pairs_batch_{batch_start}.txt'
        generate_contrastive_pairs(batch_pos_ratings, item_set, user_embeddings, item_embeddings,
                                   user_index_old2new, output_file, max_pos=max_pos)

    print('converting rating file ...')
    write_ratings(user_pos_ratings, user_neg_ratings, item_set, user_index_old2new)
    merge_files(f'../data/{DATASET}', f'../data/{DATASET}/contrastive_pairs.txt')


def write_ratings(user_pos_ratings, user_neg_ratings, item_set, user_index_old2new):
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    for user_index_old, pos_item_set in user_pos_ratings.items():
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))

        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]

        if len(unwatched_set) > 0:
            neg_items = np.random.choice(list(unwatched_set), size=min(len(pos_item_set), len(unwatched_set)),
                                         replace=False)
            for item in neg_items:
                writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % len(user_pos_ratings))
    print('number of items: %d' % len(item_set))


def compute_user_similarity(user_embeddings, top_k=None, chunk_size=1000):

    num_users = user_embeddings.shape[0]
    similarity_matrix = dok_matrix((num_users, num_users), dtype=np.float32)

    for start in range(0, num_users, chunk_size):
        end = min(start + chunk_size, num_users)
        chunk_embeddings = user_embeddings[start:end]
        chunk_similarities = cosine_similarity(chunk_embeddings, user_embeddings)

        for i, user_similarities in enumerate(chunk_similarities):
            if top_k is not None:
                top_k_indices = np.argpartition(-user_similarities, top_k)[:top_k]
            else:
                top_k_indices = np.argsort(-user_similarities)  # 完整排序，保留所有相似用户
            for idx in top_k_indices:
                if idx != start + i:  
                    similarity_matrix[start + i, idx] = user_similarities[idx]

    return similarity_matrix.tocsr()


def generate_contrastive_pairs(user_pos_ratings, item_set, user_embeddings, item_embeddings,
                               user_index_old2new, output_file, top_k=10, batch_size=500, max_pos=None):

    print('Generating contrastive pairs with dynamic parameter adjustment...')



    dataset_hard_ratio = {
        'movie1M': 0.2,
        'music': 0.1,
        'book': 0.0
    }
    hard_ratio = dataset_hard_ratio.get(DATASET, 0.0)  # 如果DATASET不在字典中，默认返回0.0
    positive_pairs = set()
    negative_pairs = set()

    user_similarity_matrix = compute_user_similarity(user_embeddings, top_k=top_k)

    batch_users = list(user_pos_ratings.keys())

    user_pos_counts = {}

    for batch_start in range(0, len(batch_users), batch_size):
        batch_end = min(batch_start + batch_size, len(batch_users))
        batch_user_pos_ratings = {u: user_pos_ratings[u] for u in batch_users[batch_start:batch_end]}

        for user, pos_items in batch_user_pos_ratings.items():
            mapped_user_index = user_index_old2new[user]
            user_embedding = user_embeddings[mapped_user_index]
            pos_items = list(pos_items)

            pos_combinations = [(item, other_item) for i, item in enumerate(pos_items) for other_item in
                                pos_items[i + 1:]]

 
            if len(pos_combinations) > max_pos:
                sampled_indices = np.random.choice(len(pos_combinations), size=max_pos, replace=False)
                pos_combinations = [pos_combinations[i] for i in sampled_indices]

            positive_pairs.update([(mapped_user_index, item1, item2) for item1, item2 in pos_combinations])


            user_pos_counts[user] = len(pos_combinations)


            current_pos = user_pos_counts[user]
            if current_pos < max_pos:
                required_cross_pos = max_pos - current_pos
                cross_user_ratio = required_cross_pos / max_pos
            else:
                required_cross_pos = 0
                cross_user_ratio = 0.0  

           
            similarities = user_similarity_matrix[mapped_user_index].toarray().flatten()
            top_k_similar_users = np.argsort(-similarities)[:top_k]
            max_cross_pairs = int(required_cross_pos * cross_user_ratio)
            cross_user_pos_count = 0
            for sim_user in top_k_similar_users:
                if sim_user in user_pos_ratings and cross_user_pos_count < max_cross_pairs:
                    sim_user_items = list(user_pos_ratings[sim_user])
                    sim_user_items_filtered = list(set(sim_user_items) - set(pos_items))
                    for item in pos_items:
                        if cross_user_pos_count >= max_cross_pairs:
                            break
                        positive_pairs.update(
                            [(mapped_user_index, item, sim_item) for sim_item in sim_user_items_filtered if
                             item != sim_item]
                        )
                        cross_user_pos_count += len(sim_user_items_filtered)


            num_neg_samples = 2 * max_pos

            negative_items = list(item_set - set(pos_items))

            if len(negative_items) > 0:
                item_distances = np.dot(user_embedding, item_embeddings[negative_items].T)
                hard_neg_indices = np.argsort(item_distances)[:int(num_neg_samples * hard_ratio)]
                hard_neg_samples = [negative_items[i] for i in hard_neg_indices]
            else:
                hard_neg_samples = []

            random_neg_sample_size = max(0, min(num_neg_samples - len(hard_neg_samples), len(negative_items)))
            random_neg_samples = np.random.choice(negative_items, size=random_neg_sample_size,
                                                  replace=False) if random_neg_sample_size > 0 else []

            mixed_neg_samples = list(set(hard_neg_samples + list(random_neg_samples)))
            negative_pairs.update(
                [(mapped_user_index, item, neg_item) for item in pos_items for neg_item in mixed_neg_samples if
                 neg_item != item]
            )

        print(f'Processed batch {batch_start // batch_size + 1}/{(len(batch_users) - 1) // batch_size + 1}')

    print(f'Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs.')

    with open(output_file, 'w', encoding='utf-8') as writer:
        for p_pair in positive_pairs:
            writer.write('%d\t%d\t%d\t1\n' % p_pair)
        for n_pair in negative_pairs:
            writer.write('%d\t%d\t%d\t0\n' % n_pair)

    print(f'Contrastive pairs saved to {output_file}')


def merge_files(output_dir, final_output_file):
    with open(final_output_file, 'w', encoding='utf-8') as outfile:
        for filename in glob.glob(f'{output_dir}/contrastive_pairs_batch_*.txt'):
            with open(filename, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())



def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')
    for line in open('../data/' + DATASET + '/kg.txt', encoding='utf-8'):
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)

def run_pipeline(dataset: str, dim: int = 64):
    np.random.seed(555)

    global args, DATASET, entity_id2index, relation_id2index, item_index_old2new
    class Args: pass
    args = Args()
    args.d = dataset
    args.dim = dim
    DATASET = dataset


    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='music', help='which dataset to preprocess')
    parser.add_argument('--dim', type=int, default=256, help='dimension of embeddings')
    args = parser.parse_args()

    run_pipeline(args.d, args.dim)