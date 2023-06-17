package org.saltedfish.myapplication

import kotlinx.serialization.Serializable

@Serializable
data class TrainData(val train_features_ids:List<IntArray>, val train_features_masks:List<IntArray>, val train_features_segments:List<IntArray> = listOf(), val train_sentiments:List<Int>)
