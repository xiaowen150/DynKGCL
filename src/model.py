import tensorflow as tf
from aggregators import SumAggregator
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np


class DynKGCL(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation):
        self.args = args
        self._parse_args(args, adj_entity, adj_relation)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation):
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
        self.pos_item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='pos_item_indices')
        self.neg_item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='neg_item_indices')
        self.contrastive_labels = tf.placeholder(dtype=tf.float32, shape=[None], name='contrastive_labels')

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=DynKGCL.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=DynKGCL.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=DynKGCL.get_initializer(), name='relation_emb_matrix')

        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)

        entities, relations = self.get_neighbors(self.item_indices)

        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)
            entity_vectors_next_iter = []

            for hop in range(self.n_iter - i):
                shape = tf.shape(entity_vectors[hop + 1])
                batch_size = tf.shape(entity_vectors[hop])[0]
                neighbor_vectors = tf.reshape(entity_vectors[hop + 1], [batch_size, -1, self.n_neighbor, self.dim])
                neighbor_relations = tf.reshape(relation_vectors[hop], [batch_size, -1, self.n_neighbor, self.dim])

                # Generate positive embeddings
                pos_vector = aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=neighbor_vectors,
                    neighbor_relations=neighbor_relations,
                    user_embeddings=self.user_embeddings,
                    sample_type='pos'
                )

                # Store the positive vectors for contrastive learning
                entity_vectors_next_iter.append(pos_vector)

            entity_vectors = entity_vectors_next_iter

        final_entity_vectors = tf.reshape(entity_vectors[0], [tf.shape(entity_vectors[0])[0], self.dim])

        return final_entity_vectors, aggregators

    def _build_contrastive_learning(self):
        pos_item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.pos_item_indices)
        neg_item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.neg_item_indices)


        pos_item_embeddings = tf.reshape(pos_item_embeddings, [-1, self.dim])
        neg_item_embeddings = tf.reshape(neg_item_embeddings, [-1, self.dim])
        user_embeddings = tf.reshape(self.user_embeddings, [-1, self.dim])


        pos_similarity = tf.reduce_sum(tf.multiply(user_embeddings, pos_item_embeddings), axis=1)
        neg_similarity = tf.reduce_sum(tf.multiply(user_embeddings, neg_item_embeddings), axis=1)

    
        safe_log_sigmoid = tf.log(tf.clip_by_value(tf.sigmoid(pos_similarity - neg_similarity), 1e-10, 1.0))
        self.contrastive_loss = -tf.reduce_mean(safe_log_sigmoid)
        self.contrastive_optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.contrastive_loss)

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.reshape(self.labels, tf.shape(self.scores)), logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        for aggregator in self.aggregators:
            self.l2_loss += tf.nn.l2_loss(aggregator.weights)


        alpha = self.args.contrastive_weight  

        self._build_contrastive_learning()

     
        self.loss = self.base_loss + self.l2_weight * self.l2_loss + alpha * self.contrastive_loss
        
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



    def train(self, sess, feed_dict, contrastive=False):
        if contrastive:
            return sess.run([self.contrastive_optimizer, self.contrastive_loss], feed_dict)
        else:
            return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        if np.isnan(scores).any():
            print("模型输出中检测到 NaN！")


        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
