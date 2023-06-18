package org.saltedfish.myapplication.TrainingTasks

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlin.reflect.KClass

@Serializable
data class ResnetTrainData(val x:List<IntArray> , val y:List<IntArray>):TrainData
class Resnet(batchSize:Int=1,dataSize:Int=1,numThreads: Int =4):TrainingTask<ResnetTrainData>(batchSize,dataSize,numThreads){
    override val TAG: String = "ResnetTraining"
    override val typeOfTrainData: KClass<ResnetTrainData> = ResnetTrainData::class
}