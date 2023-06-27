package org.saltedfish.myapplication.TrainingTasks

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.saltedfish.trainingdatahelper.TrainData
import org.saltedfish.trainingdatahelper.TrainDataClass
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import kotlin.reflect.KClass

@Serializable
@TrainDataClass
data class ResnetTrainData(val x:List<List<List<FloatArray>>> , val y:List<IntArray>): TrainData

class Resnet(batchSize:Int=1,dataSize:Int=1,numThreads: Int =4):TrainingTask<ResnetTrainData>(batchSize,dataSize,numThreads){
    override val TAG: String = "ResnetTraining"
    override val typeOfTrainData: KClass<ResnetTrainData> = ResnetTrainData::class
    override val isLazy: Boolean = true
    override val inputDimension: List<IntArray> = listOf(intArrayOf(8,10),intArrayOf(8,224,224,3))
    companion object{
        val imageProcessor= ImageProcessor.Builder()
            .add( ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .build()
    }
    override fun supplyData(
        batchSize: Int,
        batchIndex: Int,
        mapInfo: MutableMap<String, Any>
    ): MutableMap<String, Any> {
        mapInfo["imageProcessor"] = imageProcessor
        return super.supplyData(batchSize, batchIndex, mapInfo)
    }
}