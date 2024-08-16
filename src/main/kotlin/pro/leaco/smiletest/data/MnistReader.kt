package pro.leaco.smiletest.data

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.GZIPInputStream

object MnistReader {
    @Throws(IOException::class)
    fun getLabels(labelsFile: Path): IntArray {
        val bb = ByteBuffer.wrap(decompress(Files.readAllBytes(labelsFile)))
        if (bb.getInt() != 2049) {
            throw IOException("not a labels file")
        }

        val numLabels = bb.getInt()
        val labels = IntArray(numLabels)

        for (i in 0 until numLabels) {
            labels[i] = bb.get().toInt() and 0xFF
        }
        return labels
    }

    @Throws(IOException::class)
    fun getImages(imagesFile: Path): List<Array<IntArray?>> {
        val bb = ByteBuffer.wrap(decompress(Files.readAllBytes(imagesFile)))
        if (bb.getInt() != 2051) {
            throw IOException("not an images file")
        }

        val numImages = bb.getInt()
        val numRows = bb.getInt()
        val numColumns = bb.getInt()
        val images: MutableList<Array<IntArray?>> = ArrayList()

        for (i in 0 until numImages) {
            val image = arrayOfNulls<IntArray>(numRows)
            for (row in 0 until numRows) {
                image[row] = IntArray(numColumns)
                for (col in 0 until numColumns) {
                    image[row]!![col] = bb.get().toInt() and 0xFF
                }
            }
            images.add(image)
        }

        return images
    }

    @Throws(IOException::class)
    private fun decompress(input: ByteArray): ByteArray {
        ByteArrayInputStream(input).use { bais ->
            GZIPInputStream(bais).use { gis ->
                ByteArrayOutputStream().use { out ->
                    val buf = ByteArray(8192)
                    var n: Int
                    while ((gis.read(buf).also { n = it }) > 0) {
                        out.write(buf, 0, n)
                    }
                    return out.toByteArray()
                }
            }
        }
    }

    fun renderImage(image: Array<IntArray>): String {
        val sb = StringBuilder()
        val threshold1 = 256 / 3
        val threshold2 = 2 * threshold1

        for (element in image) {
            sb.append("|")
            for (pixelVal in element) {
                if (pixelVal == 0) {
                    sb.append(" ")
                } else if (pixelVal < threshold1) {
                    sb.append(".")
                } else {
                    if (pixelVal < threshold2) {
                        sb.append("x")
                    } else {
                        sb.append("X")
                    }
                }
            }
            sb.append("|\n")
        }

        return sb.toString()
    }
}
