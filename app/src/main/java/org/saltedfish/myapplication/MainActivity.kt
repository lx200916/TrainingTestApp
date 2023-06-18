package org.saltedfish.myapplication

import android.content.Context
import android.graphics.BitmapFactory
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
import org.saltedfish.myapplication.TrainingTasks.Resnet
import org.saltedfish.myapplication.ui.theme.MyApplicationTheme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.charset.Charset
import java.util.Collections

const val BATCHSIZE = 8
const val DATASIZE = 40


class MainActivity : ComponentActivity() {
    val TAG = "TFLite App"
    lateinit var assetsList: List<String>
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val resnet = Resnet(batchSize = BATCHSIZE, dataSize = DATASIZE)
        resnet.setupModel(this, dataFileName = "")
        assetsList = assets.list("pic")?.asList()?:listOf()
        resnet.registerDataSupplier { batchSize, _, map ->
            val imageProcessor = map["imageProcessor"] as ImageProcessor
            var bb = ByteBuffer.allocate(4 * batchSize * 224 * 224 * 3)
            assetsList.take(batchSize).forEach {
                var tensorImage = TensorImage(DataType.FLOAT32)
//                Assets read bitmap
                val bitmap = assets.open("pic/$it").use { input ->
                    BitmapFactory.decodeStream(input)
                }
                tensorImage.load(bitmap)
               val buffer = imageProcessor.process(tensorImage).buffer
                buffer.rewind()
                bb.put(buffer)
            }

            mutableMapOf(Pair("x",bb),Pair("y", Collections.nCopies(8,Collections.nCopies(10,1f).toFloatArray()).toTypedArray()))
         }
        resnet.startTrain()
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