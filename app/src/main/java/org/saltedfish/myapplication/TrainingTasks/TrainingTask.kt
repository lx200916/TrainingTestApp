package org.saltedfish.myapplication.TrainingTasks

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.serialization.InternalSerializationApi
import kotlinx.serialization.json.Json
import kotlinx.serialization.serializer
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.IOException
import java.nio.FloatBuffer
import java.nio.charset.Charset
import kotlin.reflect.KClass
import kotlin.reflect.KMutableProperty
import kotlin.reflect.full.declaredMemberProperties

interface TrainData {
}


abstract class TrainingTask<T : TrainData>(
    var batchSize: Int = 1,
    var dataSize: Int = 1,
    val numThreads: Int = 4
) {
    lateinit var trainData: T
    private lateinit var interpreter: Interpreter
    abstract val TAG: String
    abstract val typeOfTrainData: KClass<T>


    open fun setupModel(context: Context, modelFileName: String = "model.tflite",dataFileName: String = "data.json"){
        val options = Interpreter.Options().apply { numThreads = 4 }
        try {
            val modelFile = FileUtil.loadMappedFile(context, modelFileName)
            interpreter = Interpreter(modelFile, options)
            log("TFLite $modelFileName model loaded successfully")
            getTrainData(context,dataFileName)
        } catch (e: IOException) {
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
    }

    open fun startTrain() {
        log("${dataSize / batchSize} $dataSize $batchSize ${dataSize.floorDiv(batchSize)}")
        val batches: MutableList<MutableMap<String,Any>> = mutableListOf()
        for (i in 1..dataSize.floorDiv(batchSize)) {
            val batch = mutableMapOf<String,Any>()
            trainData::class.declaredMemberProperties.forEach {
                val value = it.getter.call(trainData)
                if (value is IntArray) {
                    val batchValue = IntArray(batchSize)
                    for (j in 0 until batchSize) {
                        batchValue[j] = value[(i - 1) * batchSize + j]
                    }
                    batch[it.name] = batchValue
                } else if (value is List<*>) {
                    val batchValue = mutableListOf<Any>()
                    batchValue.addAll(listOf(value.subList((i - 1) * batchSize, i * batchSize)))
                    batch[it.name] = batchValue
                }
            }
            batches.add(batch)
        }
        log(batches)
        GlobalScope.launch(Dispatchers.Default){
            val timeComsumed = mutableListOf<Long>()
            for (batch in batches){
                val inputs = batch
                val outputs = mutableMapOf<String, Any>()
                val loss = FloatBuffer.allocate(1)
                outputs["loss"] = loss
                val startTime = System.currentTimeMillis()
                interpreter.runSignature(inputs, outputs, "train")
                val endTime = System.currentTimeMillis()
                Log.i(TAG,"loss: ${loss.array()[0]}")
                Log.i(TAG,"Time Consumed: ${endTime-startTime}ms")
                timeComsumed.add(endTime-startTime)
            }
            if (timeComsumed.size < 3){
                Log.i(TAG,"Too few runs!")
                return@launch
            }
            timeComsumed.removeAt(0)
            Log.i(TAG,"Average Time Consumed: ${timeComsumed.average()}ms")
        }

    }


    @OptIn(InternalSerializationApi::class)
    open fun getTrainData(context: Context, dataFileName: String = "data.json") {
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

    public open fun modifyData(block: (T) -> Unit = {}) {
        block(trainData)
    }

    open fun log(vararg messages: Any) {
        messages.map {
            it.toString()
        }.joinToString { " " }.let {
            Log.i(TAG, it)
        }
    }
}