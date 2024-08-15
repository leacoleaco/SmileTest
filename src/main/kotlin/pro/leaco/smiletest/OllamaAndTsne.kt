package org.example.pro.leaco.smiletest

import io.github.amithkoujalgi.ollama4j.core.OllamaAPI
import smile.manifold.tsne
import java.awt.Desktop
import java.io.File

object OllamaAndTsne {
    @JvmStatic
    fun main(args: Array<String>) {
        val host = "http://localhost:11434/"
        val api = OllamaAPI(host)
        val model = "sam4096/qwen2tools"
        api.setRequestTimeoutSeconds(60)

        val sentences = listOf(
            "我是谁",
            "谁是我",
            "我在哪里",
            "我",
            "Who am I",
            "me",
            "where am I",
            "I"
        )
        val data = sentences.map {
            api.generateEmbeddings(
                "bge-large:latest",
                it
            ).toDoubleArray()
        }.toTypedArray()

        val start = System.currentTimeMillis()
        val tsne = tsne(data, 2, 20.0, 200.0, 550)
        val end = System.currentTimeMillis()
        System.out.format("t-SNE takes %.2f seconds\n", (end - start) / 1000.0)

        println("Cost: " + tsne.cost())


        val coordinates = tsne.coordinates


        //绘制到echart，且每个点都要显示原来的文本
        val html = """
        <html>
        <head>
            <script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.1.2/echarts.min.js"></script>
        </head>
        <body>
            <div id="main" style="width: 600px;height:400px;"></div>
            <script>
                var myChart = echarts.init(document.getElementById('main'));
                var data = [${
            coordinates.mapIndexed { index, it -> "[${it[0]}, ${it[1]}, '${sentences[index].replace("'", "\\'")}']" }
                .joinToString(",")
        }];
                var option = {
                    xAxis: {},
                    yAxis: {},
                    tooltip: {},
                    toolbox: {
                      right: 20,
                      feature: {
                        dataZoom: {}
                      }
                    },
                    dataZoom: [
                          {
                            type: 'inside'
                          },
                          {
                            type: 'slider',
                            showDataShadow: false,
                            handleIcon:
                              'path://M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
                            handleSize: '80%'
                          },
                          {
                            type: 'inside',
                            orient: 'vertical'
                          },
                          {
                            type: 'slider',
                            orient: 'vertical',
                            showDataShadow: false,
                            handleIcon:
                              'path://M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
                            handleSize: '80%'
                          }
                        ],
                    animation: false,
                    series: [{
                        type: 'scatter',
                        data: data,
                        symbolSize: 10,
                        itemStyle: {
                            color: 'blue'
                        },
                        label: {
                            show: true,
                            formatter: function (params) {
                                return params.value[2];
                            }
                        }
                    }]
                };
                myChart.setOption(option);
            </script>
        </body>
        </html>
    """.trimIndent()

        File("tsne.html")
            .also {
                it.writeText(html)
                Desktop.getDesktop().open(it)
            }
    }
}
