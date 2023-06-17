package org.saltedfish.myapplication

import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import org.json.JSONObject
import org.saltedfish.myapplication.ui.theme.MyApplicationTheme
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.IOException
import java.nio.FloatBuffer
import java.nio.charset.Charset

const val BATCHSIZE = 8
const val DATASIZE = 40


class MainActivity : ComponentActivity() {
    val TAG = "TFLite App"
    private lateinit var interpreter: Interpreter
    lateinit  var trainData:TrainData
    private fun setupModelPersonalization(context: Context): Boolean {
        val options = Interpreter.Options()
        options.numThreads = 4
        return try {
            try {
                val  trainFile= assets.open("tokenizer.json")
                val size = trainFile.available()
                val buffer = ByteArray(size)
                trainFile.read(buffer)
                trainFile.close()
                val json = String(buffer, Charset.defaultCharset())
                trainData = Json.decodeFromString<TrainData>(json)
                Log.i(TAG, "TFLite Data loaded successfully")
                Log.i(TAG,trainData.train_sentiments.size.toString())

            } catch (ex: IOException) {
                ex.printStackTrace()
            }
            val modelFile = FileUtil.loadMappedFile(context, "model.tflite")
            interpreter = Interpreter(modelFile, options)
            true
        } catch (e: IOException) {
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
            false
        }
    }
    private fun startTrain(){
        val Batches:MutableList<TrainData> = mutableListOf()
        Log.i(TAG,"${DATASIZE/BATCHSIZE} $DATASIZE $BATCHSIZE ${DATASIZE.floorDiv(BATCHSIZE)}")
        for (i in 1 ..DATASIZE.floorDiv(BATCHSIZE)){
//        for (i in 1 ..5){
            val trainBatch = TrainData(
                trainData.train_features_ids.subList((i-1)*BATCHSIZE,i*BATCHSIZE),
                trainData.train_features_masks.subList((i-1)*BATCHSIZE,i*BATCHSIZE),
                if (!trainData.train_features_segments.isEmpty()) trainData.train_features_segments.subList((i-1)*BATCHSIZE,i*BATCHSIZE) else listOf(),
                trainData.train_sentiments.subList((i-1)*BATCHSIZE,i*BATCHSIZE)
            )
            Batches.add(trainBatch)
        }
        Log.i(TAG,Batches.size.toString())

        GlobalScope.launch(Dispatchers.Default){
            val timeComsumed = mutableListOf<Long>()
            for (batch in Batches){
                val inputFeature0 = batch.train_features_ids.toTypedArray()
                val inputFeature1 = batch.train_features_masks.toTypedArray()
                val inputFeature2 = batch.train_features_segments.toTypedArray()
                val inputFeature3 = batch.train_sentiments.toIntArray()
//                val inputs = mapOf<String,Any>(Pair("input_word_ids",inputFeature0),Pair("input_mask",inputFeature1),Pair("input_type_ids",inputFeature2),Pair("y",inputFeature3))
                val inputs = mapOf<String,Any>(Pair("bert_input_ids",inputFeature0),Pair("bert_input_masks",inputFeature1),Pair("y",inputFeature3))

                val outputs: MutableMap<String, Any> = HashMap()
                val loss = FloatBuffer.allocate(1)
                outputs["loss"] = loss
//                Calc Time Consumed
                val startTime = System.currentTimeMillis()
                interpreter.runSignature(inputs, outputs, "train")
                val endTime = System.currentTimeMillis()
                Log.i(TAG,"loss: ${loss.array()[0]}")
                Log.i(TAG,"Time Consumed: ${endTime-startTime}ms")
                timeComsumed.add(endTime-startTime)
            }
            timeComsumed.removeAt(0)
            Log.i(TAG,"Average Time Consumed: ${timeComsumed.average()}ms")
        }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupModelPersonalization(this)
        startTrain()
        setContent {
            MyApplicationTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Greeting("Android")
                }
            }
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    MyApplicationTheme {
        Greeting("Android")
    }
}