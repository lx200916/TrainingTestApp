package org.saltedfish.trainingdatahelper

import kotlin.reflect.KClass


interface TrainData {
}

 open class TrainingDataHelper<T:TrainData> {
   open fun batchTrainData(trainData: Any, batchSize:Int, batchIndex: Int):MutableMap<String,Any>{
         return mutableMapOf()
   }
   companion object {}
}
