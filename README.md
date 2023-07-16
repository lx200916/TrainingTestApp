<div align="center">

  
# TFLite On-Device Training Sample

Sample On-Device Training (Benchmark) App based on TFLite. Just Define your Data and Task then FASTEN YOUR SEATBELT.
</div>

## 🚀 Get Started
1. Define Your Input Tensor Data.
   *use List<> for batches.*
```kotlin
   @Serializable
   @TrainDataClass
    data class DistilBertTrainData(val bert_input_ids:List<IntArray>, val bert_input_masks:List<IntArray>, val y:List<Int>):
    TrainData{}
```
2. Define Your Task.
```kotlin
class DistilbertTrainingTask(batchSize:Int=1,dataSize:Int=1,numThreads: Int =4):TrainingTask<DistilBertTrainData>(batchSize,dataSize,numThreads) {
    override val TAG: String = "DistilbertTrainingTask" //TAG for Logcat Output
    override val typeOfTrainData: KClass<DistilBertTrainData> = DistilBertTrainData::class // KClass of TrainData.
    override val isLazy: Boolean = false // Get Input from Assets or `Get by Lazy` with DataSupplier?(use registerDataSupplier to register a Callback.)
}
```
3. Run!
```kotlin
        val bert = DistilbertTrainingTask(batchSize = BATCHSIZE, dataSize = DATASIZE)
        bert.setupModel(this, dataFileName = "BertTokenizer.json", modelFileName = "reberta_seq_128.tflite")
        bert.startTrain()
```

## How it works?
The hardest part is to split input data into batches according to batch-size, which means convert List<*> to *[] (List<Int> to IntArray(int[]) for example).Seems quite hard to be approached by Reflect due to  `Type-Erasure` System. 
Thus CodeGen in compile-time seems a much more reasonable way.
Thanks to Kotlin KSP, we scan all the Kotlin Class with annotation `@TrainData` , then generate codes with correct property type accordingly.

## ❤️ Thanks to
* [KSP](https://github.com/google/ksp)
* [KotlinPoet](https://github.com/square/kotlinpoet)
* [Moshi](https://github.com/square/moshi)
