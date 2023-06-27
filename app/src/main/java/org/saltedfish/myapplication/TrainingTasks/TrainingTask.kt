package org.saltedfish.myapplication.TrainingTasks

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.serialization.InternalSerializationApi
import kotlinx.serialization.json.Json
import kotlinx.serialization.serializer
import org.saltedfish.trainingdatahelper.TrainData
import org.saltedfish.trainingdatahelper.TrainingDataHelper
import org.saltedfish.trainingdatahelper.gen.getHelper
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.IOException
import java.nio.FloatBuffer
import java.nio.charset.Charset
import kotlin.reflect.KClass
import kotlin.reflect.full.declaredMemberProperties
abstract class TrainingTask<T : TrainData>(
    var batchSize: Int = 1,
    var dataSize: Int = 1,
    val numThreads: Int = 4
) {
    lateinit var trainData: T
    private lateinit var interpreter: Interpreter
    protected var block: ((Int, Int, Map<String, Any>) -> MutableMap<String, Any>)? = null
    abstract val TAG: String
    abstract val typeOfTrainData: KClass<T>
    abstract val isLazy: Boolean
    open val inputDimension:List<IntArray> = listOf()
    open val trainSignature: String = "train"
    open fun setupModel(
        context: Context,
        modelFileName: String = "model.tflite",
        dataFileName: String = "data.json"
    ) {
        val options = Interpreter.Options().apply { numThreads = 4 }
        try {
            val modelFile = FileUtil.loadMappedFile(context, modelFileName)
            interpreter = Interpreter(modelFile, options)
            log("TFLite $modelFileName model loaded successfully")
            getTrainData(context, dataFileName)
        } catch (e: IOException) {
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
    }

    open fun startTrain() {
        if (!this::trainData.isInitialized) {
            if (isLazy)
                log("No TrainData! Use Callback to supply.")
            else {
                Log.e(TAG, "No TrainData!")
                return
            }
        }
        log("${dataSize / batchSize} $dataSize $batchSize ${dataSize.floorDiv(batchSize)}")

        GlobalScope.launch(Dispatchers.Default) {
            val timeComsumed = mutableListOf<Long>()
            for (i in 1..dataSize.floorDiv(batchSize)) {
                var batch = mutableMapOf<String, Any>()
                batch = if (!isLazy) {
                   batch = TrainingDataHelper<T>().getHelper(typeOfTrainData)?.batchTrainData(
                       trainData,batchSize, i-1
                   )?:batch
                    Log.i(TAG,"Batch: ${batch.size}")
                    supplyData(batchSize, i, batch)
                } else
                    supplyData(batchSize, i)
                val outputs = mutableMapOf<String, Any>()
                val loss = FloatBuffer.allocate(1)
                outputs["loss"] = loss
                println(batch)
                if (batch.any {
                        return@any !it.value.javaClass.isArray&&!it.value.javaClass.name.contains("ist")}){
                    Log.i(TAG,"Find Buffer  reallocate the buffer")
                    inputDimension.forEachIndexed { index, ints ->
                        interpreter.resizeInput(index, ints)
                    }
                    interpreter.allocateTensors()
//                    Log.i(TAG,"Buffer reallocated ${interpreter.getInputTensor(1).shapeSignature().joinToString()}")
                }
                val startTime = System.currentTimeMillis()
                interpreter.runSignature(batch, outputs, trainSignature)
                val endTime = System.currentTimeMillis()
                Log.i(TAG, "loss: ${loss.array()[0]}")
                Log.i(TAG, "Time Consumed: ${endTime - startTime}ms")
                timeComsumed.add(endTime - startTime)
            }
            if (timeComsumed.size < 3) {
                Log.i(TAG, "Too few runs!")
                return@launch
            }
            timeComsumed.removeAt(0)
            Log.i(TAG, "Average Time Consumed: ${timeComsumed.average()}ms")
        }

    }


    @OptIn(InternalSerializationApi::class)
    open fun getTrainData(context: Context, dataFileName: String = "data.json") {
        if (dataFileName.isEmpty()) return
        try {
            val trainFile = context.assets.open(dataFileName)
            val size = trainFile.available()
            val buffer = ByteArray(size)
            trainFile.read(buffer)
            trainFile.close()
            val json = String(buffer, Charset.defaultCharset())
            trainData = Json { ignoreUnknownKeys = true }.decodeFromString<T>(
                typeOfTrainData.serializer(),
                json
            )
            log("TFLite Data loaded successfully")
            trainData::class.declaredMemberProperties.forEach {
                log(it.name)
                val value = it.getter.call(trainData)
                if (value is IntArray) {
                    if (value.size < dataSize) {
                        log("dataSize is larger than the size of the data")
                        dataSize = value.size
                    }
                } else if (value is List<*>) {
                    if (value.size < dataSize) {
                        log("dataSize is larger than the size of the data")
                        dataSize = value.size
                    }
                }
            }

        } catch (ex: IOException) {
            ex.printStackTrace()
        }

    }

    open fun modifyData(block: (T) -> Unit = {}) {
        block(trainData)
    }

    open fun registerDataSupplier(block: ((Int, Int, Map<String, Any>) -> MutableMap<String, Any>)? = null) {
        this.block = block
    }

    open fun supplyData(
        batchSize: Int,
        batchIndex: Int,
        mapInfo: MutableMap<String, Any> = mutableMapOf()
    ): MutableMap<String, Any> {
        if (block == null) Log.e(TAG, "No Data Supply Func!")
        return block?.invoke(batchSize, batchIndex, mapInfo)
            ?: if (isLazy) mutableMapOf() else mapInfo
    }

    open fun log(vararg messages: Any) {
        messages.map {
            it.toString()
        }.joinToString { " " }.let {
            Log.i(TAG, it)
        }
    }
}