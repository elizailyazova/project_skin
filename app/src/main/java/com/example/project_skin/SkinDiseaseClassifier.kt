package com.example.project_skin

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class SkinDiseaseClassifier(modelPath: String, assetManager: AssetManager) {

    private val interpreter: Interpreter

    init {
        val model = loadModelFile(assetManager, modelPath)
        interpreter = Interpreter(model)
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }


    fun classifyImage(bitmap: Bitmap): String {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val byteBuffer = convertBitmapToByteBuffer(resizedBitmap)

        val inputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        inputBuffer.loadBuffer(byteBuffer)

        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, NUM_CLASSES), DataType.FLOAT32)

        Log.d("SkinClassifier", "Input tensor shape: [1, 224, 224, 3]")
        Log.d("SkinClassifier", "Output tensor shape: [1, ${NUM_CLASSES}]")

        interpreter.run(inputBuffer.buffer, outputBuffer.buffer)

        val probabilities = outputBuffer.floatArray
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1

        Log.d("SkinClassifier", "Model probabilities: ${probabilities.joinToString()}")
        return if (probabilities[maxIndex] < 0.5) "Healthy" else CLASS_NAMES[maxIndex]
    }
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true) 
        Log.d("SkinClassifier", "Resized bitmap dimensions: ${resizedBitmap.width}x${resizedBitmap.height}")

        val byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3) 
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(224 * 224)
        resizedBitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224)

        for (pixel in intValues) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }

        Log.d("SkinClassifier", "ByteBuffer size: ${byteBuffer.capacity()} bytes")
        return byteBuffer
    }

    companion object {
        private val CLASS_NAMES = arrayOf("Acne", "Keratosis", "Carcinoma", "Rosacea", "Eczema")
        private const val NUM_CLASSES = 5
    }
}
