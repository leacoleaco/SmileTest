package pro.leaco.smiletest

import pro.leaco.smiletest.data.MnistReader.getImages
import pro.leaco.smiletest.data.MnistReader.getLabels
import smile.feature.extraction.pca
import smile.manifold.TSNE
import smile.math.MathEx
import smile.plot.swing.ScatterPlot
import java.nio.file.Paths


object MinistWithTsne {
    @JvmStatic
    fun main(args: Array<String>) {

        val projectDir = System.getProperty("user.dir")

        val path = "${projectDir}/src/main/resources/data/minist"

        val label = getLabels(Paths.get(path, "train-labels-idx1-ubyte.gz"))
        val images = getImages(Paths.get(path, "train-images-idx3-ubyte.gz"))

        val filter = images
            .asSequence()
            .mapIndexed { index, arr ->
                arr to label[index]
            }
            .take(2500)


        val x =
            filter.map { it.first }.map { x -> x.flatMap { y -> y!!.map { it.toDouble() } }.toDoubleArray() }.toList()
                .toTypedArray()
        val y = filter.map { it.second }.toList().toIntArray()

        val pca = pca(x)
        //拿到前50个主成分
        val projection = pca.getProjection(50)
        //将数据投影到前50个主成分上
        val x50 = projection.apply(x)
        //NOTE: 注意一定要先归一化, 否则t-SNE 计算结果为 NaN
        MathEx.normalize(x50)

        x50.forEachIndexed { index, array ->
            check(!array.any { it.isNaN() || it.isInfinite() }) {
                "Invalid data found in ${index} imagesArr : $array"
            }
        }


        //t-SNE降维, 降到2维，条件分布的困惑度为 20.0, 学习率为200 ,迭代次数为1000
        //see: https://haifengl.github.io/manifold.html
        val tsne = TSNE(x50, 3, 20.0, 200.0, 1000)
        val canvas = ScatterPlot.of(tsne.coordinates, y, '@').canvas();
        canvas.setTitle("t-SNE of MNIST");
        canvas.window();
    }
}
