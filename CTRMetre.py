# contrastive learning prototype
            projection = inputs[-1]

            projection_1 = tf.layers.dense(
                inputs=projection,
                units=64,
                kernel_initializer=tf.glorot_uniform_initializer(seed=123),
                bias_initializer=tf.zeros_initializer(),
                name = 'projection_1')

            projection_1 = tf.layers.batch_normalization(
                inputs=projection_1)

            projection_2 = tf.nn.relu(projection_1)

            projection_2 = tf.layers.dense(
                inputs=projection_2,
                units=32,
                kernel_initializer=tf.glorot_uniform_initializer(seed=123),
                bias_initializer=tf.zeros_initializer(),
                name = 'projection_2')

            projection_2 = tf.layers.batch_normalization(
                inputs=projection_2)

            projection_2 = tf.nn.relu(projection_2)

            projection_2_normalize = tf.math.l2_normalize(projection_2, axis=1, name='projection_norm')

            projection_layer = tf.layers.Dense(
                units=num_domains_tower,
                use_bias=False,
                kernel_initializer=tf.truncated_normal_initializer(stddev=1e-4, seed=4),
                name='prototype')

            prototype = projection_layer(projection_2_normalize)

            prototype = tf.layers.batch_normalization(
                inputs=prototype)

        outputs = []
        for i in range(num_domains_tower):
            dc1 = tf.layers.dense(
                inputs=inputs[i],
                units=32,
                activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(seed=123),
                bias_initializer=tf.zeros_initializer())

            output = tf.layers.dense(
                inputs=dc1,
                units=2,
                kernel_initializer=tf.glorot_uniform_initializer(seed=123),
                bias_initializer=tf.zeros_initializer())
            outputs.append(output[:input_batch_size])

    tf.logging.info('trainable_variables {}'.format(tf.trainable_variables()))

# clustering probabilty
        crops_for_assign = [0, 1]
        Q = tf.zeros_like(prototype[:input_batch_size], dtype = tf.float32)
        with tf.GradientTape() as tape:
            for i, crop_id in enumerate(crops_for_assign): # crops_for_assign = [0, 1]
                with tape.stop_recording():
                    out = prototype[input_batch_size * crop_id: input_batch_size * (crop_id + 1)]
                    q = sinkhorn(out)
                    Q += q
                    subloss = 0
                    import numpy as np
                    for v in np.delete(np.arange(2), crop_id):
                        temperature = 0.1
                        p = tf.nn.softmax(prototype[input_batch_size * v: input_batch_size * (v + 1)] / temperature)
                        logp = tf.math.log(p)
                        subloss -= tf.math.reduce_mean(tf.math.reduce_sum(q * logp , axis=1))
                        subloss -= tf.math.reduce_mean(tf.math.reduce_sum(pes_prob * logp , axis=1))
                contrastive_loss += subloss / tf.cast(1, tf.float32)