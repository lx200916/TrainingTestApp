package org.saltedfish.trainingdatahelper
/*
*  TrainDataClass Annotation marks the Data Class as a TrainData Class so we generate TrainingDataHelper code for it
*/
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.SOURCE)
public annotation class TrainDataClass