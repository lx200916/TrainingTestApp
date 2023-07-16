package org.saltedfish.myapplication.TrainingTasks

import kotlinx.serialization.Serializable
import org.saltedfish.trainingdatahelper.TrainData
import org.saltedfish.trainingdatahelper.TrainDataClass
import kotlin.reflect.KClass

@Serializable
@TrainDataClass
data class DistilBertTrainData(val bert_input_ids:List<IntArray>, val bert_input_masks:List<IntArray>, val y:List<Int>):
    TrainData{}
class DistilbertTrainingTask(batchSize:Int=1,dataSize:Int=1,numThreads: Int =4):TrainingTask<DistilBertTrainData>(batchSize,dataSize,numThreads) {
    override val TAG: String = "DistilbertTrainingTask"
    override val typeOfTrainData: KClass<DistilBertTrainData> = DistilBertTrainData::class
    override val isLazy: Boolean = false
}