package org.saltedfish.myapplication.TrainingTasks
import kotlinx.serialization.Serializable
import org.saltedfish.trainingdatahelper.TrainData
import org.saltedfish.trainingdatahelper.TrainDataClass
import kotlin.reflect.KClass

@Serializable
@TrainDataClass
data class BertTrainData(val input_word_ids:List<IntArray>, val input_mask:List<IntArray>,val input_type_ids:List<IntArray>, val y:List<Int>):
    TrainData{}
class BertTrainingTask(batchSize:Int=1,dataSize:Int=1,numThreads: Int =4):TrainingTask<BertTrainData>(batchSize,dataSize,numThreads) {
    override val TAG: String = "BertTrainingTask"
    override val typeOfTrainData: KClass<BertTrainData> = BertTrainData::class
    override val isLazy: Boolean = false
}
