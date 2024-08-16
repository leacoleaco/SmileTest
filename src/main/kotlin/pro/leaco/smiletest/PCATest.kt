package pro.leaco.smiletest

import pro.leaco.smiletest.data.MnistReader.getImages
import pro.leaco.smiletest.data.MnistReader.getLabels
import smile.feature.extraction.pca
import smile.plot.swing.ScatterPlot
import smile.plot.swing.ScreePlot
import java.nio.file.Paths

object PCATest {
    @JvmStatic
    fun main(args: Array<String>) {

        val projectDir = System.getProperty("user.dir")

        val path = "${projectDir}/src/main/resources/data/minist"

        val label = getLabels(Paths.get(path, "train-labels-idx1-ubyte.gz"))
        val images = getImages(Paths.get(path, "train-images-idx3-ubyte.gz"))

        val filter = images
            .asSequence()
            .mapIndexedNotNull { index, arr ->
                arr to label[index]
            }
            .take(2500)


        val x =
            filter.map { it.first }.map { x -> x.flatMap { y -> y!!.map { it.toDouble() } }.toDoubleArray() }.toList()
                .toTypedArray()
        val y = filter.map { it.second }.toList().toIntArray()

        //PCA降维
        val pca = pca(x)

        //绘制每个成分的方差比例图
        ScreePlot(pca.varianceProportion()).canvas().window()

        //拿到前三个主成分
        val projection = pca.getProjection(3)

        //将数据投影到前三个主成分上
        val x2 = projection.apply(x)

        //绘制散点图, 使用“*” 作为散点的形状
        ScatterPlot.of(x2, y, '*').canvas().window();


    }
}
