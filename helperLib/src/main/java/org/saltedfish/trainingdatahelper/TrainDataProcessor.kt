package org.saltedfish.trainingdatahelper

import com.google.devtools.ksp.containingFile
import com.google.devtools.ksp.processing.CodeGenerator
import com.google.devtools.ksp.processing.KSPLogger
import com.google.devtools.ksp.processing.Resolver
import com.google.devtools.ksp.processing.SymbolProcessor
import com.google.devtools.ksp.processing.SymbolProcessorEnvironment
import com.google.devtools.ksp.processing.SymbolProcessorProvider
import com.google.devtools.ksp.symbol.KSAnnotated
import com.google.devtools.ksp.symbol.KSClassDeclaration
import com.google.devtools.ksp.symbol.KSFile
import com.google.devtools.ksp.symbol.KSType
import com.google.devtools.ksp.validate
import com.squareup.kotlinpoet.ClassName
import com.squareup.kotlinpoet.CodeBlock
import com.squareup.kotlinpoet.FileSpec
import com.squareup.kotlinpoet.FunSpec
import com.squareup.kotlinpoet.KModifier
import com.squareup.kotlinpoet.ParameterSpec
import com.squareup.kotlinpoet.ParameterizedTypeName.Companion.parameterizedBy
import com.squareup.kotlinpoet.TypeSpec
import com.squareup.kotlinpoet.WildcardTypeName
import com.squareup.kotlinpoet.asClassName
import com.squareup.kotlinpoet.asTypeName
import com.squareup.kotlinpoet.ksp.toClassName
import com.squareup.kotlinpoet.ksp.writeTo
import kotlin.reflect.KClass
class TrainDataProcessor(
    val codeGenerator: CodeGenerator,
    val logger: KSPLogger
) : SymbolProcessor {
    var invoked = false
    val allSymbolsSet:MutableSet<ClassName> = mutableSetOf()
    override fun process(resolver: Resolver): List<KSAnnotated> {
        val files = mutableListOf<KSFile>()
        val allSymbols = resolver.getSymbolsWithAnnotation(TrainDataClass::class.java.name)
        val result = allSymbols.filter { !it.validate() }
        val symbols = allSymbols.filter { it.validate() && it is KSClassDeclaration }
        logger.warn("Generating Training Data Helper for ${allSymbols.count()} classes")
        symbols.forEach {
            files.add(it.containingFile!!)
            val classDeclaration = it as KSClassDeclaration
            logger.warn("Generating Training Data Helper for ${classDeclaration.simpleName.asString()}")
            val fileSpec = FileSpec.builder("org.saltedfish.trainingdatahelper.gen", "${classDeclaration.simpleName.asString()}helper")
            fileSpec.addType(
                TypeSpec.classBuilder("${classDeclaration.simpleName.asString()}Helper")
                    .superclass(
                        TrainingDataHelper::class.asClassName()
                            .parameterizedBy(classDeclaration.toClassName())
                    )
                    .addFunction(
                        FunSpec.builder("batchTrainData").addModifiers(KModifier.OVERRIDE)
                            // return MutableMap<String,Any>
                            .returns(
                                ClassName("kotlin.collections", "MutableMap").parameterizedBy(
                                    String::class.asTypeName(),
                                    Any::class.asTypeName()
                                )
                            )
                            .addParameters(
                                listOf(
                                    ParameterSpec("trainData", Any::class.asTypeName()),
                                    ParameterSpec("batchSize", Int::class.asClassName()),
                                    ParameterSpec("batchIndex", Int::class.asClassName())
                                )
                            ).addCode(CodeBlock.builder().let { builder ->
                                val properties = classDeclaration.getAllProperties()
                                builder.addStatement("val map = mutableMapOf<String,Any>()")
                                builder.addStatement("trainData as? ${classDeclaration.toClassName()}?:return map\n")
                                properties.forEach {
                                    builder.addStatement("val p${it.simpleName.asString()} = trainData.${it.simpleName.asString()}")
                                    //Check if the property is a list
                                    if (it.type.resolve().declaration.qualifiedName?.asString()
                                            ?.contains("collection") == true
                                    ) {
                                        builder.addStatement(
                                            "val p${it.simpleName.asString()}Batch = p${it.simpleName.asString()}.subList(batchIndex*batchSize,(batchIndex+1)*batchSize).${
                                                getReturnFunc(
                                                    it.type.resolve()
                                                    
                                                )
                                            }"
                                        )
                                    } else {
                                        builder.addStatement("val p${it.simpleName.asString()}Batch =  Array(batchSize) {p${it.simpleName.asString()}}")
                                    }
                                    builder.addStatement("map[\"${it.simpleName.asString()}\"] = p${it.simpleName.asString()}Batch")

                                }
                                builder.addStatement("return map")

                                builder.build()
                            })
                            .build()
                    )
                    .build()

            )
            allSymbolsSet.add(classDeclaration.toClassName())
            fileSpec.build().writeTo(codeGenerator, true)
        }

        return result.toList()
    }

    override fun finish() {
        val fileSpec = FileSpec.builder("org.saltedfish.trainingdatahelper.gen", "TrainingDataHelperExt")
        fileSpec.addFunction(FunSpec.builder("getHelper").receiver(
            ClassName(
                "org.saltedfish.trainingdatahelper",
                "TrainingDataHelper"
            ).parameterizedBy(WildcardTypeName.producerOf(TrainData::class.asTypeName()))
        )
            .addParameter(
                "type",
                KClass::class.asTypeName()
                    .parameterizedBy(WildcardTypeName.producerOf(Any::class.asTypeName()))
            )
            .returns(
                TrainingDataHelper::class.asTypeName()
                    .parameterizedBy(WildcardTypeName.producerOf(TrainData::class.asTypeName()))
                    .copy(true)
            )

            .addCode("return when(type){").let { builder ->

                allSymbolsSet.forEach {
                    val classDeclaration = it
                    fileSpec.addImport(it, "")
                    builder.addStatement("${classDeclaration.simpleName}::class -> ${classDeclaration.simpleName}Helper()")
                }
                builder.addStatement("else -> null")
                builder.addStatement("}")
                builder.build()
            })
        fileSpec.build().writeTo(codeGenerator, true)
//        logger.warn("Finish!")
    }

}

fun getReturnFunc(type: KSType): String {
    return when (type.arguments.firstOrNull()?.type.toString()) {
        "Int" -> "toIntArray()"
        "Long" -> "toLongArray()"
        "Float" -> "toFloatArray()"
        "Double" -> "toDoubleArray()"
        "Char" -> "toCharArray()"
        "Byte" -> "toByteArray()"
        "Short" -> "toShortArray()"
        "Boolean" -> "toBooleanArray()"
        else -> "toTypedArray()"
    }
}

class TrainDataProcessorProvider : SymbolProcessorProvider {
    override fun create(environment: SymbolProcessorEnvironment): SymbolProcessor {
        return TrainDataProcessor(environment.codeGenerator, environment.logger)
    }
}